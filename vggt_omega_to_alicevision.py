#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT-Omega  ->  AliceVision  :  end-to-end textured-mesh reconstruction.

    images/  --[VGGT-Omega]-->  camera poses + dense depth
             --[this script]->  AliceVision SfMData (.sfm) + depthMap/simMap (.exr)
                                + color-harmonised texture images
             --[aliceVision_meshing]-------> Delaunay tetrahedralisation + graph cut
             --[aliceVision_meshFiltering]-> smoothed / largest-component mesh
             --[aliceVision_meshDecimate]--> (optional) simplified mesh
             --[aliceVision_texturing]-----> texturedMesh.obj + .mtl + texture_*.png

Why this works
--------------
`aliceVision_meshing` does not care where depth maps come from.  It reads per-view
`<viewId>_depthMap.exr` (+ optional `<viewId>_simMap.exr`) out of one folder plus an
SfMData file holding the cameras.  So we write VGGT-Omega's predictions in exactly
that layout and hand it to the native binaries.

Conventions that have to be exact (all verified against a real AliceVision cache):

1.  AliceVision depth is the DISTANCE ALONG THE VIEWING RAY:
        backproject(cam, (x, y), d) = C + normalize(iCam * (x, y, 1)) * d
    VGGT-Omega predicts Z-DEPTH.  Conversion:
        d = z * || ((x - cx) / fx, (y - cy) / fy, 1) ||
    Both index pixels the same way (integer index, no +0.5 offset).

2.  A view's image resolution must be an EXACT INTEGER MULTIPLE k of its depth-map
    resolution.  MultiViewParams takes k from the `AliceVision:downscale` EXR
    attribute (or from `view.width / depthmap.width`, integer division) and then
    assumes `getWidth() == depthmap.width`.  We therefore emit color images at
    exactly (k*Wv, k*Hv); k defaults to whatever puts them closest to the inputs.

3.  SfMData JSON is boost::property_tree; `saveMatrix` serialises Eigen matrices by
    LINEAR index and Eigen's Matrix3d is COLUMN-MAJOR, so
    `poses[i].pose.transform.rotation` is column-major.  Intrinsics are stored as
    focal-length-in-mm + pixel ratio + principal-point-offset-from-image-center.

4.  AliceVision REFUSES an SfMData whose version is newer than the binary
    ("File has a version more recent than this library").  We therefore declare
    1.2.6 by default, which every AliceVision from 2023 onwards accepts, and use
    that version's focal-length convention.  Use --sfm-version 1.2.14 only if you
    are on a recent build and want the newer semantics.

Skin-tone continuity
--------------------
Blotchy skin on a textured face is almost always per-view exposure / white-balance
drift, not a texturing bug.  Measured on real 57-view face captures the median skin
L* varies by 24-34 units across views.  `--harmonize gain` applies a per-view
von-Kries gain in linear RGB so every view matches the median view before AliceVision
blends them; on the same captures that collapses the L* range to < 1 and roughly
halves the b* range.  Combined with AliceVision's multi-band blending
(--multi-band-nb-contrib) it is the single biggest lever on seam visibility.

Requirements
------------
    torch, numpy, pillow, torchvision and the `vggt_omega` package (pip install -e .)
    OpenImageIO python bindings (preferred, writes full metadata) or opencv-python
    AliceVision binaries -- pass --av-bin <AliceVision>/bin

Examples (Ubuntu)
-----------------
    # one capture -- output defaults to .../albert_185119/images_alicevision
    python vggt_omega_to_alicevision.py \
        --images   /data/albert_185119/images \
        --masks    /data/albert_185119/bgrm/dilation/masks \
        --checkpoint ./checkpoints/vggt_omega_1b_512.pt \
        --av-bin   /opt/Meshroom-2023.3.0/aliceVision/bin

    # every capture under a root (auto-detects bgrm masks + brightness-CCT-adjust);
    # each capture still gets its own <images>_alicevision next to its images/
    python vggt_omega_to_alicevision.py \
        --batch-root /data/20250415/0415 \
        --checkpoint ./checkpoints/vggt_omega_1b_512.pt \
        --av-bin   /opt/Meshroom-2023.3.0/aliceVision/bin \
        --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SENSOR_WIDTH_MM = 36.0  # AliceVision's default virtual sensor; only the ratio matters
INVALID_DEPTH = -1.0  # AliceVision treats depth <= 0 as "no measurement"

# Folders a capture usually ships with (batch mode auto-detection)
AUTO_MASK_CANDIDATES = ["masks_david", "bgrm/dilation/masks", "bgrm/erosion/masks", "bgrm/parser/masks", "masks"]
AUTO_TEXTURE_CANDIDATES = ["brightness-CCT-adjust/dilation", "brightness-CCT-adjust/parser", "brightness_adjusted"]


def log(msg: str = "") -> None:
    print(f"[vggt2av] {msg}", flush=True)


class Section:
    def __init__(self, title: str) -> None:
        self.title = title

    def __enter__(self) -> "Section":
        log("=" * 74)
        log(self.title)
        log("=" * 74)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc: Any) -> None:
        log(f"-> {self.title}: {time.time() - self.t0:.1f}s")


# --------------------------------------------------------------------------------------
# 1. preprocessing -- mirrors vggt_omega.utils.load_fn but records the geometry
# --------------------------------------------------------------------------------------


@dataclass
class FrameGeometry:
    """Everything needed to rebuild the exact frame the network saw, at any scale."""

    path: Path
    original_size: tuple[int, int]  # (w, h) of the source file
    crop_box: tuple[int, int, int, int]  # (left, top, right, bottom) in source pixels
    net_size: tuple[int, int]  # (w, h) fed to the network, before padding
    pad: tuple[int, int, int, int]  # (left, top, right, bottom), in network pixels
    frame_size: tuple[int, int]  # (w, h) of the padded frame == network input

    mask_path: Path | None = None
    texture_path: Path | None = None  # alternative (color-corrected) source image

    # filled in after inference
    view_id: int = 0
    intrinsic: np.ndarray = field(default_factory=lambda: np.eye(3))  # at frame_size
    extrinsic: np.ndarray = field(default_factory=lambda: np.eye(3, 4))  # camera-from-world
    depth: np.ndarray | None = None  # z-depth, (h, w)
    conf: np.ndarray | None = None  # (h, w)
    gain: np.ndarray = field(default_factory=lambda: np.ones(3))  # harmonisation gain

    # filled in by cross_view_consensus(); once set these REPLACE the per-view filtering
    ray_depth_fused: np.ndarray | None = None  # AliceVision ray distance, already scaled
    valid_fused: np.ndarray | None = None


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(np.round(float(value) / multiple)) * multiple)


def _crop_box_for_supported_aspect_ratio(
    width: int, height: int, min_ar: float = 0.5, max_ar: float = 2.0
) -> tuple[int, int, int, int]:
    """load_fn._crop_to_supported_aspect_ratio, expressed as a box."""
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_ar:
        crop_w = min(width, max(1, int(round(height / min_ar))))
        left = max((width - crop_w) // 2, 0)
        return (left, 0, left + crop_w, height)
    if aspect_ratio > max_ar:
        crop_h = min(height, max(1, int(round(width * max_ar))))
        top = max((height - crop_h) // 2, 0)
        return (0, top, width, top + crop_h)
    return (0, 0, width, height)


def _target_shape(aspect_ratio: float, image_resolution: int, patch_size: int, mode: str) -> tuple[int, int]:
    """(h, w) of the network input.  Same two sizing modes as load_fn."""
    if mode == "balanced":
        token_number = (image_resolution // patch_size) ** 2
        w_patches = np.sqrt(token_number / max(aspect_ratio, 1e-9))
        h_patches = token_number / max(w_patches, 1e-9)
        return max(1, int(np.round(h_patches))) * patch_size, max(1, int(np.round(w_patches))) * patch_size
    if aspect_ratio >= 1.0:
        return image_resolution, _round_to_multiple(image_resolution / aspect_ratio, patch_size)
    return _round_to_multiple(image_resolution * aspect_ratio, patch_size), image_resolution


def _open_rgb(path: Path):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as raw:
        if raw.mode == "RGBA":
            background = Image.new("RGBA", raw.size, (255, 255, 255, 255))
            raw = Image.alpha_composite(background, raw)
        return raw.convert("RGB")


def preprocess_images(
    image_paths: Sequence[Path],
    image_resolution: int = 512,
    patch_size: int = 16,
    mode: str = "balanced",
):
    """Reproduce load_and_preprocess_images, keeping the per-frame provenance.

    Returns (tensor[S, 3, H, W] in [0, 1], list[FrameGeometry]).
    """
    import torch
    from PIL import Image
    from torchvision import transforms as TF

    to_tensor = TF.ToTensor()
    tensors: list[Any] = []
    geoms: list[FrameGeometry] = []

    for path in image_paths:
        image = _open_rgb(Path(path))
        original_size = image.size
        box = _crop_box_for_supported_aspect_ratio(*original_size)
        image = image.crop(box)
        w, h = image.size
        target_h, target_w = _target_shape(h / max(w, 1), image_resolution, patch_size, mode)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        tensors.append(to_tensor(image))
        geoms.append(
            FrameGeometry(
                path=Path(path),
                original_size=original_size,
                crop_box=box,
                net_size=(target_w, target_h),
                pad=(0, 0, 0, 0),
                frame_size=(target_w, target_h),
            )
        )

    shapes = {(t.shape[2], t.shape[1]) for t in tensors}  # (w, h)
    if len(shapes) > 1:
        max_w = max(s[0] for s in shapes)
        max_h = max(s[1] for s in shapes)
        log(f"heterogeneous frame sizes {sorted(shapes)} -> padding all to {max_w}x{max_h}")
        padded = []
        for tensor, geom in zip(tensors, geoms):
            w_pad, h_pad = max_w - tensor.shape[2], max_h - tensor.shape[1]
            pad_left, pad_top = w_pad // 2, h_pad // 2
            if w_pad or h_pad:
                tensor = torch.nn.functional.pad(
                    tensor, (pad_left, w_pad - pad_left, pad_top, h_pad - pad_top), mode="constant", value=1.0
                )
            geom.pad = (pad_left, pad_top, w_pad - pad_left, h_pad - pad_top)
            geom.frame_size = (max_w, max_h)
            padded.append(tensor)
        tensors = padded

    return torch.stack(tensors), geoms


def render_at_scale(source: Path, geom: FrameGeometry, k: int, nearest: bool = False):
    """Rebuild the frame the network saw, at k x resolution, from a full-resolution file.

    This is what keeps the texture as sharp as the inputs allow: pixels come from the
    source JPEG, not from an upsampled 512 px tensor.
    """
    from PIL import Image

    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    image = _open_rgb(source)
    if image.size != geom.original_size:
        raise ValueError(
            f"{source} is {image.size} but the matching input image is {geom.original_size}; "
            "auxiliary folders must be pixel-aligned with --images"
        )

    image = image.crop(geom.crop_box)
    net_w, net_h = geom.net_size
    image = image.resize((net_w * k, net_h * k), resample)

    if any(geom.pad):
        pad_l, pad_t, _, _ = geom.pad
        frame_w, frame_h = geom.frame_size
        canvas = Image.new("RGB", (frame_w * k, frame_h * k), (255, 255, 255))
        canvas.paste(image, (pad_l * k, pad_t * k))
        image = canvas
    return image


def frame_padding_mask(geom: FrameGeometry) -> np.ndarray:
    """False on the white padding bands (nothing was ever observed there)."""
    frame_w, frame_h = geom.frame_size
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    pad_l, pad_t, _, _ = geom.pad
    net_w, net_h = geom.net_size
    mask[pad_t : pad_t + net_h, pad_l : pad_l + net_w] = True
    return mask


def subject_mask_at_depth_scale(geom: FrameGeometry) -> np.ndarray | None:
    """Foreground mask resampled into the depth-map (network frame) grid."""
    if geom.mask_path is None:
        return None
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(geom.mask_path) as raw:
        mask = raw.convert("L")
    if mask.size != geom.original_size:
        raise ValueError(f"mask {geom.mask_path} is {mask.size}, expected {geom.original_size}")
    mask = mask.crop(geom.crop_box).resize(geom.net_size, Image.Resampling.NEAREST)
    array = np.asarray(mask) > 127

    frame_w, frame_h = geom.frame_size
    out = np.zeros((frame_h, frame_w), dtype=bool)
    pad_l, pad_t, _, _ = geom.pad
    net_w, net_h = geom.net_size
    out[pad_t : pad_t + net_h, pad_l : pad_l + net_w] = array
    return out


# --------------------------------------------------------------------------------------
# 2. inference
# --------------------------------------------------------------------------------------


def run_vggt_omega(
    geoms: list[FrameGeometry], images, checkpoint: Path, device: str
) -> list[FrameGeometry]:
    import torch

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().eval()
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict and "aggregator.patch_embed.proj.weight" not in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)

    with torch.inference_mode():
        predictions = model(images.to(device))

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])

    extrinsic = extrinsic[0].detach().float().cpu().numpy()  # (S, 3, 4), camera-from-world
    intrinsic = intrinsic[0].detach().float().cpu().numpy()  # (S, 3, 3)
    depth = predictions["depth"][0].detach().float().cpu().numpy()  # (S, H, W, 1)
    conf = predictions["depth_conf"][0].detach().float().cpu().numpy()  # (S, H, W)
    if depth.ndim == 4:
        depth = depth[..., 0]

    for i, geom in enumerate(geoms):
        geom.extrinsic = extrinsic[i].astype(np.float64)
        geom.intrinsic = intrinsic[i].astype(np.float64)
        geom.depth = depth[i].astype(np.float32)
        geom.conf = conf[i].astype(np.float32)

    del model, predictions
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return geoms


def resolve_device(requested: str, required: bool = True) -> str:
    """Choose a usable PyTorch device and fail early for an invalid request."""
    try:
        import torch
    except ImportError:
        if required:
            raise
        # --skip-inference / --dry-run only drive the AliceVision binaries; torch is
        # not needed and may not even be installed on the machine doing the meshing
        return requested.strip().lower()

    requested = requested.strip().lower()
    if requested == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"device: auto -> {selected}")
        return selected
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {requested!r} was requested, but this PyTorch installation has no usable CUDA device. "
            "Use --device cpu, or install a CUDA-enabled PyTorch build."
        )
    return requested


# --------------------------------------------------------------------------------------
# 3. geometry conversions
# --------------------------------------------------------------------------------------


def zdepth_to_ray_distance(z_depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """VGGT z-depth -> AliceVision distance-along-ray."""
    h, w = z_depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xs = (np.arange(w, dtype=np.float64)[None, :] - cx) / fx
    ys = (np.arange(h, dtype=np.float64)[:, None] - cy) / fy
    return (z_depth.astype(np.float64) * np.sqrt(xs**2 + ys**2 + 1.0)).astype(np.float32)


def depth_edge_mask(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    """True where the local relative depth jump exceeds rtol (flying pixels, silhouettes)."""
    pad = kernel_size // 2
    padded = np.pad(depth, pad, mode="edge")
    h, w = depth.shape
    dmax = np.full_like(depth, -np.inf)
    dmin = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[y : y + h, x : x + w]
            np.maximum(dmax, window, out=dmax)
            np.minimum(dmin, window, out=dmin)
    return (dmax - dmin) / np.maximum(np.abs(depth), 1e-6) > rtol


def confidence_to_similarity(conf: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """VGGT confidence (>=1, higher is better) -> AliceVision simMap in [-1, 0].

    AliceVision weights a fused point by `1 + (1 + sim) * simFactor`, so sim = -1 is the
    best possible measurement.  We normalise confidence between its 10th and 90th
    percentile over the valid pixels.
    """
    sim = np.zeros_like(conf, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(conf[valid], [10.0, 90.0])
        hi = max(hi, lo + 1e-6)
        sim = (-np.clip((conf - lo) / (hi - lo), 0.0, 1.0)).astype(np.float32)
    sim[~valid] = 0.0
    return sim


def scaled_intrinsic(K: np.ndarray, k: int, aligned: bool = False) -> np.ndarray:
    """Network-resolution K -> K of the k-times larger texture image / depth grid.

    aligned=False (legacy, v1-v3): plain diag(k, k, 1) @ K.
    aligned=True: also shifts the principal point by (k-1)/2.  Every resize in this
    pipeline (PIL for the photos, cv2 for the depth maps) is centre-aligned, so network
    pixel i covers image pixels k*i .. k*i+k-1 and its centre lands on k*i + (k-1)/2, not
    on k*i.  Without the shift each view's rays -- depth AND colour -- are off by
    (k-1)/2 image pixels (1 px at k=3, ~0.13 mm on the face) in that view's own image
    axes, so different views disagree with each other by up to that much.
    """
    Ks = np.diag([float(k), float(k), 1.0]) @ K
    if aligned:
        Ks[0, 2] += 0.5 * (k - 1)
        Ks[1, 2] += 0.5 * (k - 1)
    return Ks


def unscaled_intrinsic(K_img: np.ndarray, k: int, aligned: bool = False) -> np.ndarray:
    """Inverse of scaled_intrinsic()."""
    K = K_img.copy()
    if aligned:
        K[0, 2] -= 0.5 * (k - 1)
        K[1, 2] -= 0.5 * (k - 1)
    return np.diag([1.0 / k, 1.0 / k, 1.0]) @ K


def projection_matrix(K: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Row-major 4x4 P = [K [R|t]; 0 0 0 1] -- the AliceVision:P metadata layout."""
    P = np.eye(4, dtype=np.float64)
    P[:3, :4] = K @ extrinsic[:3, :4]
    return P


def _view_validity(geom: FrameGeometry, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """(valid, ray_depth) for one view -- the same filtering export_frames applies."""
    if geom.ray_depth_fused is not None and geom.valid_fused is not None:
        return geom.valid_fused, geom.ray_depth_fused
    z_depth, conf = geom.depth, geom.conf
    valid = frame_padding_mask(geom) & np.isfinite(z_depth) & (z_depth > 0)
    subject = subject_mask_at_depth_scale(geom)
    if subject is not None:
        valid &= subject
    if args.conf_percentile > 0 and valid.any():
        valid &= conf >= float(np.percentile(conf[valid], args.conf_percentile))
    if args.conf_min > 0:
        valid &= conf >= args.conf_min
    if args.depth_edge_rtol > 0:
        valid &= ~depth_edge_mask(z_depth, rtol=args.depth_edge_rtol)
    ray_depth = zdepth_to_ray_distance(z_depth, geom.intrinsic) * args.scene_scale
    return valid, ray_depth


# --------------------------------------------------------------------------------------
# cross-view consensus -- the step that makes VGGT depth maps meshable
# --------------------------------------------------------------------------------------
#
# Measured on patient_27_20260511_143743 (9 views), reprojecting every view into every
# other and comparing against that view's own depth, in units of pixSize (= depth / focal,
# the world size of one depth-map pixel -- the unit AliceVision's fusion margins use):
#
#     depth maps fed to aliceVision_meshing        median |Δd|   within 2·pixSize
#     -------------------------------------------  -----------   ----------------
#     VGGT-Omega raw, res512 (k=7)                     3.51            33.8 %
#     VGGT-Omega raw, res1024 (k=3)                    5.14            23.9 %
#     Meshroom SGM raw                                22.20            32.7 %
#     Meshroom AFTER DepthMapFilter  <-- the target    0.36            76.1 %
#
# Meshroom's raw SGM maps are even worse than VGGT's, but its DepthMapFilter node throws
# away every pixel that fewer than `minNumOfConsistentCams` other views corroborate
# (84% -> 29% of pixels survive), so the mesher only ever sees sub-pixel-consistent data.
# Our pipeline writes VGGT maps straight into the "filtered" slot, so the graph cut gets
# 9 mutually-offset surface sheets and faithfully carves the gaps between them -- which is
# exactly the branching "dried mud" crack network seen on the reconstructed faces.
#
# Single-view depth is NOT the problem: VGGT's per-view |laplacian| is 0.24 pixSize versus
# 0.15 for Meshroom's filtered maps.  The disagreement is purely between views.
#
# This function does what DepthMapFilter does, plus one extra step: instead of only
# deleting disagreeing pixels it replaces each pixel with the MEDIAN of the corroborating
# views, which actively pulls the sheets together.  Measured effect (3 passes + smoothing):
#
#     median inter-view |Δd|   3.51 -> 0.56 pixSize      (target 0.36)
#     within 2·pixSize         33.8% -> 72.6%            (target 76.1%)
#     per-view |laplacian|     0.24 -> 0.06 pixSize      (target 0.15)
#     valid pixels             41.4% -> 33.7%            (Meshroom keeps 29.1%)


def _depth_cam(geom: FrameGeometry, scene_scale: float) -> dict[str, Any]:
    """Camera at DEPTH-MAP resolution (geom.intrinsic is already at that scale)."""
    R = geom.extrinsic[:3, :3]
    t = geom.extrinsic[:3, 3]
    K = geom.intrinsic
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "R": R,
        "C": (-R.T @ t) * scene_scale,
        "axis": R[2],  # optical axis in world space
    }


def _unit_rays(shape: tuple[int, int], cam: dict[str, Any]) -> np.ndarray:
    h, w = shape
    xs = (np.arange(w, dtype=np.float64)[None, :] - cam["cx"]) / cam["fx"]
    ys = (np.arange(h, dtype=np.float64)[:, None] - cam["cy"]) / cam["fy"]
    v = np.stack([np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w)), np.ones((h, w))], -1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _backproject(depth: np.ndarray, cam: dict[str, Any], rays: np.ndarray | None = None) -> np.ndarray:
    if rays is None:
        rays = _unit_rays(depth.shape, cam)
    return cam["C"] + (rays @ cam["R"]) * depth[..., None]  # R^T @ v == v @ R


def _project(points: np.ndarray, cam: dict[str, Any]):
    local = (points - cam["C"]) @ cam["R"].T
    z = local[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = local[..., 0] / z * cam["fx"] + cam["cx"]
        v = local[..., 1] / z * cam["fy"] + cam["cy"]
    return u, v, z


def _neighbour_views(cams: list[dict[str, Any]], index: int, count: int, max_angle_deg: float) -> list[int]:
    """The `count` views whose optical axis is closest to this one, within max_angle."""
    axis = cams[index]["axis"]
    scored = []
    for j, other in enumerate(cams):
        if j == index:
            continue
        cosine = float(np.clip(axis @ other["axis"], -1.0, 1.0))
        angle = np.degrees(np.arccos(cosine))
        if angle <= max_angle_deg:
            scored.append((angle, j))
    scored.sort()
    return [j for _, j in scored[:count]]


def _edge_aware_smooth(depth: np.ndarray, valid: np.ndarray, fx: float, radius: int, tol: float) -> np.ndarray:
    """Box average over neighbours that are within `tol` pixSize of the centre pixel."""
    if radius <= 0:
        return depth
    pix = np.where(valid, depth, np.nan) / fx
    accumulator = np.zeros_like(depth)
    count = np.zeros_like(depth)
    threshold = tol * np.nan_to_num(pix, nan=np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(depth, dy, 0), dx, 1)
            shifted_ok = np.roll(np.roll(valid, dy, 0), dx, 1)
            take = shifted_ok & valid & (np.abs(shifted - depth) < threshold)
            accumulator[take] += shifted[take]
            count[take] += 1
    return np.where(count > 0, accumulator / np.maximum(count, 1), depth)


def cross_view_consensus(geoms: Sequence[FrameGeometry], args: argparse.Namespace) -> None:
    """Make the per-view depth maps agree with each other before they reach the mesher."""
    if args.consensus_passes <= 0:
        log("cross-view consensus: disabled (--consensus-passes 0)")
        return

    cams = [_depth_cam(g, args.scene_scale) for g in geoms]
    rays = [_unit_rays(g.depth.shape, c) for g, c in zip(geoms, cams)]

    depths, valids = [], []
    for geom in geoms:
        valid, ray_depth = _view_validity(geom, args)
        depth = ray_depth.astype(np.float64).copy()
        depth[~valid] = INVALID_DEPTH
        depths.append(depth)
        valids.append(valid.copy())

    neighbours = [
        _neighbour_views(cams, i, args.consensus_neighbours, args.consensus_max_view_angle)
        for i in range(len(geoms))
    ]
    before = 100.0 * float(np.mean([v.mean() for v in valids]))

    bilinear = getattr(args, "consensus_sampling", "nearest") == "bilinear"
    fuse = getattr(args, "consensus_fuse", "median")

    for pass_index in range(args.consensus_passes):
        updated = []
        surfaces: dict[int, np.ndarray] = {}  # each view's own surface, backprojected once per pass
        for rc in range(len(geoms)):
            depth_rc, valid_rc = depths[rc], valids[rc]
            if not valid_rc.any():
                updated.append((depth_rc, valid_rc))
                continue
            points = _backproject(depth_rc, cams[rc], rays[rc])
            world_rays = rays[rc] @ cams[rc]["R"]
            pix_rc = depth_rc / cams[rc]["fx"]

            stack = [np.where(valid_rc, depth_rc, np.nan)]
            agreeing = np.zeros(depth_rc.shape, dtype=np.int32)

            for tc in neighbours[rc]:
                depth_tc, valid_tc = depths[tc], valids[tc]
                height, width = depth_tc.shape
                u, v, z = _project(points, cams[tc])
                inside = valid_rc & (z > 0) & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
                candidate = np.full_like(depth_rc, np.nan)
                if inside.any():
                    if tc not in surfaces:
                        surfaces[tc] = _backproject(depth_tc, cams[tc], rays[tc])
                    yi, xi = np.nonzero(inside)
                    uf, vf = u[inside], v[inside]
                    ui = np.round(uf).astype(np.int32)
                    vi = np.round(vf).astype(np.int32)
                    hit = valid_tc[vi, ui]
                    if hit.any():
                        yi, xi, ui, vi, uf, vf = yi[hit], xi[hit], ui[hit], vi[hit], uf[hit], vf[hit]
                        # tc's own surface point, re-expressed as a depth along rc's ray
                        surface = surfaces[tc][vi, ui]
                        if bilinear:
                            # nearest-pixel lookup quantises the neighbour's surface to its
                            # pixel grid: on a sloped cheek that is up to +-0.5 px of
                            # position error turned into a depth error that aliases into a
                            # regular ripple.  Interpolate where all four corners are valid
                            # and belong to the same surface.
                            x0 = np.floor(uf).astype(np.int32)
                            y0 = np.floor(vf).astype(np.int32)
                            ax = (uf - x0)[:, None]
                            ay = (vf - y0)[:, None]
                            corners = [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]
                            ok4 = np.logical_and.reduce([valid_tc[cy, cx] for cy, cx in corners])
                            dz = np.stack([depth_tc[cy, cx] for cy, cx in corners], 0)
                            ok4 &= (dz.max(0) - dz.min(0)) < args.consensus_tolerance * pix_rc[yi, xi]
                            s00, s01, s10, s11 = (surfaces[tc][cy, cx] for cy, cx in corners)
                            interp = (1 - ay) * ((1 - ax) * s00 + ax * s01) + ay * ((1 - ax) * s10 + ax * s11)
                            surface = np.where(ok4[:, None], interp, surface)
                        along_ray = np.einsum("ij,ij->i", surface - cams[rc]["C"], world_rays[yi, xi])
                        keep = np.abs(along_ray - depth_rc[yi, xi]) < args.consensus_tolerance * pix_rc[yi, xi]
                        candidate[yi[keep], xi[keep]] = along_ray[keep]
                        agreeing[yi[keep], xi[keep]] += 1
                stack.append(candidate)

            with warnings.catch_warnings():  # all-NaN columns are expected (invalid pixels)
                warnings.simplefilter("ignore", RuntimeWarning)
                stacked = np.stack(stack, 0)
                fused = np.nanmedian(stacked, axis=0)
                if fuse == "inlier-mean":
                    # The median picks ONE view's value per pixel (or the mean of two), and
                    # which view that is changes from pixel to pixel -> a patchwork of
                    # 0.5-pixSize steps.  Averaging every corroborating value close to the
                    # median is smoother and, for Gaussian noise, ~1.5x more efficient.
                    band = args.consensus_fuse_band * pix_rc
                    inl = np.abs(stacked - fused[None]) <= band[None]
                    fused = np.where(
                        inl.any(0), np.nansum(np.where(inl, stacked, 0.0), 0) / np.maximum(inl.sum(0), 1), fused
                    )
            new_valid = valid_rc & (agreeing >= args.consensus_min_agree) & np.isfinite(fused)
            fused = np.where(new_valid, fused, INVALID_DEPTH)
            updated.append((fused, new_valid))

        depths = [d for d, _ in updated]
        valids = [v for _, v in updated]
        kept = 100.0 * float(np.mean([v.mean() for v in valids]))
        log(f"  consensus pass {pass_index + 1}/{args.consensus_passes}: valid pixels {kept:.1f}%")

    for geom, cam, depth, valid in zip(geoms, cams, depths, valids):
        if args.consensus_smooth_radius > 0:
            depth = _edge_aware_smooth(
                depth, valid, cam["fx"], args.consensus_smooth_radius, args.consensus_smooth_tolerance
            )
        depth = depth.astype(np.float32)
        depth[~valid] = INVALID_DEPTH
        geom.ray_depth_fused = depth
        geom.valid_fused = valid

    after = 100.0 * float(np.mean([v.mean() for v in valids]))
    log(
        f"cross-view consensus: {args.consensus_passes} passes, tol {args.consensus_tolerance} pixSize, "
        f"minAgree {args.consensus_min_agree}, {args.consensus_neighbours} neighbours "
        f"-> valid pixels {before:.1f}% -> {after:.1f}%"
    )


def build_dense_seed_landmarks(
    geoms: Sequence[FrameGeometry], k: int, args: argparse.Namespace, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Multi-view landmarks synthesised from VGGT-Omega's own depth+pose.

    Not for the mesh itself -- these only seed aliceVision_depthMapEstimation:

    * MultiViewParams::findNearestCamsFromLandmarks picks each view's neighbouring cameras
      by counting landmarks shared with >20 matching-angle observations; with zero shared
      landmarks (a single observation per point) every view gets 0 neighbours and depth
      estimation produces nothing.
    * SgmDepthList::getMinMaxMidNbDepthFromSfM needs landmarks observed in a view to seed
      that view's own depth search range.

    A 3D point sampled from one view's depth is kept as an additional observation of
    another view only where it reprojects, within tolerance, onto that other view's own
    (independently predicted) depth -- i.e. only where VGGT-Omega's per-view predictions
    already agree with each other, which is the same cross-view consistency a real
    triangulated SfM point would have.
    """
    per_view: dict[int, dict[str, Any]] = {}
    for geom in geoms:
        valid, ray_depth = _view_validity(geom, args)
        R, t = geom.extrinsic[:3, :3], geom.extrinsic[:3, 3] * args.scene_scale
        center = -R.T @ t
        frame_w, frame_h = geom.frame_size
        per_view[geom.view_id] = {
            "valid": valid,
            "ray_depth": ray_depth,
            "K": geom.intrinsic,
            "R": R,
            "t": t,
            "center": center,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }

    tol = args.dense_landmark_depth_tol
    landmarks: list[dict[str, Any]] = []

    for geom in geoms:
        rc = geom.view_id
        vr = per_view[rc]
        ys, xs = np.nonzero(vr["valid"])
        if xs.size == 0:
            continue
        n = min(args.dense_landmarks_per_view, xs.size)
        pick = rng.choice(xs.size, size=n, replace=False)
        xs, ys = xs[pick], ys[pick]
        depths = vr["ray_depth"][ys, xs].astype(np.float64)
        i_cam = vr["R"].T @ np.linalg.inv(vr["K"])
        rays = i_cam @ np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
        rays /= np.linalg.norm(rays, axis=0, keepdims=True)
        points = (vr["center"][:, None] + rays * depths[None, :]).T  # (N, 3)

        off = 0.5 * (k - 1) if args.pixel_center == "aligned" else 0.0
        obs_pixels: list[dict[int, tuple[float, float]]] = [
            {rc: (float(x * k + off), float(y * k + off))} for x, y in zip(xs, ys)
        ]

        for other in geoms:
            tc = other.view_id
            if tc == rc:
                continue
            vt = per_view[tc]
            p_cam = points @ vt["R"].T + vt["t"][None, :]
            in_front = p_cam[:, 2] > 1e-6
            uvw = p_cam @ vt["K"].T
            with np.errstate(invalid="ignore", divide="ignore"):
                u = uvw[:, 0] / uvw[:, 2]
                v = uvw[:, 1] / uvw[:, 2]
            iu = np.round(u).astype(np.int64)
            iv = np.round(v).astype(np.int64)
            in_bounds = in_front & (iu >= 0) & (iu < vt["frame_w"]) & (iv >= 0) & (iv < vt["frame_h"])
            if not in_bounds.any():
                continue
            idx = np.nonzero(in_bounds)[0]
            valid_tc = vt["valid"][iv[idx], iu[idx]]
            idx = idx[valid_tc]
            if idx.size == 0:
                continue
            expected = np.linalg.norm(points[idx] - vt["center"][None, :], axis=1)
            actual = vt["ray_depth"][iv[idx], iu[idx]]
            agree = np.abs(expected - actual) < tol * expected
            for i in idx[agree]:
                obs_pixels[i][tc] = (float(u[i] * k + off), float(v[i] * k + off))

        landmarks.extend({"X": point, "observations": obs} for point, obs in zip(points, obs_pixels))

    nb_multiview = sum(1 for lm in landmarks if len(lm["observations"]) > 1)
    log(f"seed landmarks: {len(landmarks)} total, {nb_multiview} confirmed by >=2 views")
    return landmarks


def _landmarks_to_pycolmap(
    lms: list[dict[str, Any]],
    view_ids: list[int],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    min_inliers_per_frame: int,
):
    """Build a pycolmap.Reconstruction (cameras + poses + tracked points) from our own
    seed landmarks, ready for pycolmap.bundle_adjustment().

    Mirrors facebookresearch/vggt's vggt/dependency/np_to_pycolmap.py:
    batch_np_matrix_to_pycolmap, trimmed to the one case we need: masks supplied directly
    (our landmarks are already cross-view verified, so no reprojection-error prefilter),
    one PINHOLE camera per view (no shared intrinsics, no distortion).

    Returns (reconstruction, valid_idx) where valid_idx[j] is lms' index of the j-th
    (1-indexed) point3D in the reconstruction, or (None, None) if any view has too few
    inliers for pycolmap to bundle-adjust.
    """
    import pycolmap

    n = len(view_ids)
    id_to_idx = {vid: i for i, vid in enumerate(view_ids)}
    p = len(lms)

    tracks = np.zeros((n, p, 2), dtype=np.float64)
    masks = np.zeros((n, p), dtype=bool)
    for j, lm in enumerate(lms):
        for vid, (x, y) in lm["observations"].items():
            i = id_to_idx[vid]
            tracks[i, j] = (x, y)
            masks[i, j] = True

    if masks.sum(1).min() < min_inliers_per_frame:
        return None, None

    inlier_num = masks.sum(0)
    valid_idx = np.nonzero(inlier_num >= 2)[0]
    if valid_idx.size == 0:
        return None, None

    # pycolmap >=3.11 (we target 4.2.0) models every image via a rig/frame pair even for a
    # plain single-camera capture; add_camera_with_trivial_rig / add_image_with_trivial_frame
    # are the documented shortcuts that create a 1:1 rig/frame per camera/image so the rest
    # of this function can stay pose-per-image, as if it were the older flat API.
    reconstruction = pycolmap.Reconstruction()
    for vidx in valid_idx:
        reconstruction.add_point3D(lms[vidx]["X"], pycolmap.Track(), np.zeros(3))

    for fidx in range(n):
        fx, fy, cx, cy = (
            intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2],
        )
        camera = pycolmap.Camera(
            model="PINHOLE", width=int(image_size[0]), height=int(image_size[1]),
            params=np.array([fx, fy, cx, cy]), camera_id=fidx + 1,
        )
        reconstruction.add_camera_with_trivial_rig(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )
        image = pycolmap.Image(name=f"view_{view_ids[fidx]}", camera_id=camera.camera_id, image_id=fidx + 1)

        points2d = []
        for point3d_id, vidx in enumerate(valid_idx, start=1):
            if not masks[fidx, vidx]:
                continue
            points2d.append(pycolmap.Point2D(tracks[fidx, vidx], point3d_id))
            reconstruction.points3D[point3d_id].track.add_element(fidx + 1, len(points2d) - 1)

        image.points2D = pycolmap.Point2DList(points2d)
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction, valid_idx


def refine_poses_with_ba(
    geoms: Sequence[FrameGeometry], k: int, landmarks: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    """Refine VGGT-Omega's per-view poses+intrinsics with a pycolmap bundle adjustment,
    using the --dense-mvs seed landmarks as tracks (see build_dense_seed_landmarks).

    VGGT-Omega's camera head is a single feed-forward regression with no iterative
    refinement, so its reprojection error is typically several pixels -- enough to bias
    aliceVision_depthMapEstimation's per-view PatchMatch stereo (which assumes accurate
    epipolar geometry) into a noisy, "bumpy" surface even where the underlying depth is
    fine. This does not add AliceVision feature matching: the correspondences are the
    same cross-view-verified landmarks already synthesised from VGGT-Omega's own depth.

    Mutates geom.extrinsic / geom.intrinsic in place; updates each used landmark's "X" to
    its bundle-adjusted 3D position.
    """
    import pycolmap

    view_ids = [g.view_id for g in geoms]
    lms = [lm for lm in landmarks if len(lm["observations"]) >= 2]
    if len(lms) < args.ba_min_inliers_per_frame:
        log(f"BA: only {len(lms)} multiview seed landmarks, too few to refine poses -- skipping")
        return

    frame_w, frame_h = geoms[0].frame_size
    image_size = (frame_w * k, frame_h * k)
    # landmark "X" positions come out of build_dense_seed_landmarks already multiplied by
    # scene_scale (it scales translation before triangulating); match that convention here
    # so the initial reconstruction is self-consistent, then divide back out below.
    extrinsics = np.stack([g.extrinsic.astype(np.float64) for g in geoms])
    extrinsics[:, :3, 3] *= args.scene_scale
    aligned = args.pixel_center == "aligned"
    intrinsics = np.stack([scaled_intrinsic(g.intrinsic, k, aligned) for g in geoms])

    reconstruction, valid_idx = _landmarks_to_pycolmap(
        lms, view_ids, extrinsics, intrinsics, image_size, args.ba_min_inliers_per_frame
    )
    if reconstruction is None:
        log("BA: at least one view has too few multiview landmarks -- skipping pose refinement "
            "(try a lower --ba-min-inliers-per-frame or a higher --dense-landmarks-per-view)")
        return

    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    for i, geom in enumerate(geoms):
        pyimage = reconstruction.images[i + 1]
        # cam_from_world is a bound method (pose lives on the image's frame), not a property,
        # on pycolmap's rig/frame-based Reconstruction (>=3.11)
        geom.extrinsic = np.asarray(pyimage.cam_from_world().matrix(), dtype=np.float64)
        geom.extrinsic[:3, 3] /= args.scene_scale
        K = np.asarray(reconstruction.cameras[pyimage.camera_id].calibration_matrix(), dtype=np.float64)
        geom.intrinsic = unscaled_intrinsic(K, k, aligned)

    for point3d_id, vidx in enumerate(valid_idx, start=1):
        if point3d_id in reconstruction.points3D:  # BA's outlier filtering may drop a point
            lms[vidx]["X"] = np.asarray(reconstruction.points3D[point3d_id].xyz, dtype=np.float64)

    log(f"BA: refined {len(geoms)} camera poses with pycolmap "
        f"({len(valid_idx)}/{len(lms)} landmarks used as tracks)")


# --------------------------------------------------------------------------------------
# 4. color harmonisation (skin-tone continuity)
# --------------------------------------------------------------------------------------


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = a / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055) * 255.0


def survey_view_colors(geoms: Sequence[FrameGeometry], sample_width: int = 320) -> np.ndarray:
    """Median linear-RGB of every view's subject region, from cheap thumbnails."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    medians = []
    for geom in geoms:
        source = geom.texture_path or geom.path
        with Image.open(source) as raw:
            image = raw.convert("RGB")
            scale = sample_width / max(image.width, 1)
            thumb = image.resize((sample_width, max(1, int(image.height * scale))), Image.Resampling.BILINEAR)
        array = np.asarray(thumb, dtype=np.float32)

        select = np.ones(array.shape[:2], dtype=bool)
        if geom.mask_path is not None:
            with Image.open(geom.mask_path) as raw:
                mask = raw.convert("L").resize(thumb.size, Image.Resampling.NEAREST)
            select &= np.asarray(mask) > 127
        luminance = array.mean(-1)
        select &= (luminance > 40) & (luminance < 245)  # ignore clipped shadows/highlights
        if select.sum() < 200:
            select = (luminance > 40) & (luminance < 245)
        if select.sum() < 50:
            medians.append(np.array([0.5, 0.5, 0.5]))
            continue
        medians.append(np.median(srgb_to_linear(array[select]), axis=0))
    return np.asarray(medians)


def compute_harmonisation_gains(geoms: Sequence[FrameGeometry], mode: str) -> None:
    """Per-view gain that maps each view onto the median view (von Kries style)."""
    if mode == "none":
        return
    medians = survey_view_colors(geoms)
    target = np.median(medians, axis=0)
    for geom, median in zip(geoms, medians):
        gain = target / np.maximum(median, 1e-6)
        if mode == "luminance":
            gain = np.full(3, float(gain.mean()))
        geom.gain = np.clip(gain, 0.25, 4.0)

    spread = np.array([g.gain for g in geoms])
    log(
        f"harmonisation ({mode}): gain R {spread[:,0].min():.3f}-{spread[:,0].max():.3f}  "
        f"G {spread[:,1].min():.3f}-{spread[:,1].max():.3f}  B {spread[:,2].min():.3f}-{spread[:,2].max():.3f}"
    )


def apply_gain(image, gain: np.ndarray):
    from PIL import Image

    if np.allclose(gain, 1.0, atol=1e-3):
        return image
    array = np.asarray(image, dtype=np.float32)
    out = linear_to_srgb(srgb_to_linear(array) * gain[None, None, :])
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------------------
# 5. EXR output
# --------------------------------------------------------------------------------------


class ExrWriter:
    """float32 EXR writer; emits full AliceVision metadata when OpenImageIO is present."""

    def __init__(self) -> None:
        self.backend: str | None = None
        try:
            import OpenImageIO  # noqa: F401

            self.backend = "oiio"
        except Exception:
            try:
                os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
                import cv2  # noqa: F401

                self.backend = "cv2"
            except Exception:
                self.backend = None
        if self.backend is None:
            raise RuntimeError(
                "No EXR writer available. Install OpenImageIO python bindings "
                "(conda install -c conda-forge openimageio) or `pip install opencv-python`."
            )
        if self.backend == "cv2":
            log("EXR backend: opencv (metadata omitted; AliceVision falls back to the SfMData)")
        else:
            log("EXR backend: OpenImageIO")

    def write(self, path: Path, array: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        array = np.ascontiguousarray(array.astype(np.float32))
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "oiio":
            self._write_oiio(path, array, metadata or {})
        else:
            self._write_cv2(path, array)

    @staticmethod
    def _write_oiio(path: Path, array: np.ndarray, metadata: dict[str, Any]) -> None:
        import OpenImageIO as oiio

        if array.ndim == 2:
            array = array[:, :, None]
        height, width, channels = array.shape
        spec = oiio.ImageSpec(width, height, channels, "float")

        aggregates = {
            "matrix44": oiio.AGGREGATE.MATRIX44,
            "matrix33": oiio.AGGREGATE.MATRIX33,
            "vec3": oiio.AGGREGATE.VEC3,
        }
        for key, value in metadata.items():
            try:
                if isinstance(value, tuple) and len(value) == 2 and value[0] in aggregates:
                    kind, payload = value
                    spec.attribute(key, oiio.TypeDesc(oiio.BASETYPE.DOUBLE, aggregates[kind]), tuple(payload))
                elif isinstance(value, bool):
                    spec.attribute(key, int(value))
                else:
                    spec.attribute(key, value)
            except Exception as exc:  # metadata is a nice-to-have, never fatal
                log(f"  warning: could not set EXR attribute {key}: {exc}")

        out = oiio.ImageOutput.create(str(path))
        if out is None or not out.open(str(path), spec):
            raise RuntimeError(f"OpenImageIO cannot write {path}")
        out.write_image(array)
        out.close()

    @staticmethod
    def _write_cv2(path: Path, array: np.ndarray) -> None:
        import cv2

        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if not cv2.imwrite(str(path), array, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
            raise RuntimeError(f"cv2 failed to write {path}")


# --------------------------------------------------------------------------------------
# 6. SfMData writer
# --------------------------------------------------------------------------------------


def build_sfmdata(
    geoms: Sequence[FrameGeometry],
    k: int,
    image_paths: dict[int, Path],
    version: str,
    landmarks: Sequence[dict[str, Any]] | None = None,
    pixel_center_aligned: bool = False,
) -> dict[str, Any]:
    """AliceVision SfMData, boost::property_tree JSON flavour.

    `version` selects the focal-length convention:
      * "1.2.6"  -> loader calls setFocalLength(f_mm, ratio, useCompatibility=true)
                    => fx = f_mm * W / sensorWidth,  fy = fx * pixelRatio
      * "1.2.14" -> loader calls setFocalLength(f_mm, ratio)
                    => fy = f_mm * W / sensorWidth,  fx = fy / pixelRatio

    `landmarks`, if given, become the SfMData "structure" section: sparse points with a
    single observation each (see sample_landmarks -- this is a seed for AliceVision's own
    dense depth estimation, not the dense point cloud itself).
    """
    legacy = tuple(int(v) for v in version.split(".")) < (1, 2, 11)
    views, intrinsics, poses = [], [], []

    for geom in geoms:
        frame_w, frame_h = geom.frame_size
        img_w, img_h = frame_w * k, frame_h * k
        K = scaled_intrinsic(geom.intrinsic, k, pixel_center_aligned)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        pixel_ratio = fy / fx
        focal_mm = (fx if legacy else fy) * SENSOR_WIDTH_MM / img_w

        R = geom.extrinsic[:3, :3]  # camera-from-world rotation
        t = geom.extrinsic[:3, 3]
        center = -R.T @ t  # camera center in world space

        views.append(
            {
                "viewId": str(geom.view_id),
                "poseId": str(geom.view_id),
                "frameId": str(geom.view_id),
                "intrinsicId": str(geom.view_id),
                "resectionId": str(geom.view_id),
                "path": str(image_paths[geom.view_id].resolve()).replace("\\", "/"),
                "width": str(img_w),
                "height": str(img_h),
                "metadata": {"Make": "VGGT-Omega", "Model": "VGGT-Omega", "oiio:ColorSpace": "sRGB"},
            }
        )

        intrinsics.append(
            {
                "intrinsicId": str(geom.view_id),
                "width": str(img_w),
                "height": str(img_h),
                "sensorWidth": f"{SENSOR_WIDTH_MM:.10g}",
                "sensorHeight": f"{SENSOR_WIDTH_MM * img_h / img_w:.10g}",
                "serialNumber": "vggt-omega",
                "type": "pinhole",
                "initializationMode": "calibrated",
                "initialFocalLength": "-1",
                "focalLength": f"{focal_mm:.17g}",
                "pixelRatio": f"{pixel_ratio:.17g}",
                "pixelRatioLocked": "true",
                "offsetLocked": "true",
                "scaleLocked": "true",
                # stored as an OFFSET from the image center
                "principalPoint": [f"{cx - img_w / 2.0:.17g}", f"{cy - img_h / 2.0:.17g}"],
                "distortionInitializationMode": "none",
                "distortionParams": "",
                "undistortionOffset": ["0", "0"],
                "undistortionParams": "",
                "distortionType": "none",
                "undistortionType": "none",
                "locked": "true",
            }
        )

        # Eigen Matrix3d is COLUMN-major and saveMatrix uses a linear index
        poses.append(
            {
                "poseId": str(geom.view_id),
                "pose": {
                    "transform": {
                        "rotation": [f"{v:.17g}" for v in R.flatten(order="F")],
                        "center": [f"{v:.17g}" for v in center],
                    },
                    "locked": "1",
                    "rotationOnly": "false",
                    "removable": "false",
                },
            }
        )

    structure = []
    for index, landmark in enumerate(landmarks or []):
        observations = landmark["observations"]  # {view_id: (x, y)}
        structure.append(
            {
                "landmarkId": str(index),
                "referenceViewIndex": str(next(iter(observations))),
                "descType": "unknown",
                "isParallaxRobust": "false",
                "isLocked": "false",
                "color": ["128", "128", "128"],
                "X": [f"{v:.17g}" for v in landmark["X"]],
                "observations": [
                    {
                        "observationId": str(view_id),
                        "featureId": "0",
                        "x": [f"{v:.17g}" for v in xy],
                        "scale": "0",
                        "depth": "-1",
                    }
                    for view_id, xy in observations.items()
                ],
            }
        )

    sfm: dict[str, Any] = {
        "version": version.split("."),
        "views": views,
        "intrinsics": intrinsics,
        "poses": poses,
    }
    if structure:
        sfm["structure"] = structure
    return sfm


def save_sfmdata(sfm: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sfm, handle, indent=1)
    nb_landmarks = len(sfm.get("structure", []))
    suffix = f", {nb_landmarks} seed landmarks" if nb_landmarks else ""
    log(f"wrote {path}  ({len(sfm['views'])} views, sfmDataIO {'.'.join(sfm['version'])}{suffix})")


# --------------------------------------------------------------------------------------
# 7. export depth / sim maps and texture images
# --------------------------------------------------------------------------------------


def upsample_to_image_grid(
    ray_depth: np.ndarray, sim: np.ndarray, valid: np.ndarray, k: int, rtol: float, method: str = "linear"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a network-resolution depth map onto the camera's own pixel grid.

    aliceVision_meshing measures every fusion tolerance in pixSize -- the world size of ONE
    depth-map pixel (depth / focal, focal taken at the depth map's own scale).  Writing the
    maps at network resolution therefore hurts twice over: k^2 times fewer samples AND a
    pixSize k times too large, so `pixSizeMarginInitCoef=2` spans k times more world distance
    and the Delaunay fuses away detail that the data actually supports.

    Measured against a pure-Meshroom run of the same capture (patient_27_20260522_153941),
    whose Meshing node used exactly the parameters we do:

        depth maps fed to aliceVision_meshing     size         valid samples   raw mesh
        ----------------------------------------  -----------  -------------   ---------
        Meshroom DepthMapFilter (downscale 1)      3000x4000      16,187,790     327,706 v
        ours, network resolution (k=3)              880x1184       2,429,432      45,593 v

    6.7x the samples, 7.2x the vertices -- the whole gap, with identical meshing parameters.

    Bilinear interpolation must not cross a depth discontinuity: AliceVision would read the
    interpolated value as real surface and web the nose edge to the cheek behind it.  Any
    output pixel whose source neighbourhood straddles a jump larger than `rtol` (or touches
    an invalid pixel) is therefore dropped rather than interpolated.

    method="cubic" uses bicubic instead of bilinear inside continuous regions.  Bilinear
    is only C0: the upsampled surface is a grid of flat-ish bilinear patches whose normals
    jump at every network pixel (0.46 mm on the face), and nine views laid on top of each
    other at different orientations print that grid into the mesh as a fine orange-peel
    relief.  Bicubic is C1 across cells.  Its 4x4 support is one pixel wider, so the guard
    band around jumps / invalid pixels is widened to match.
    """
    if k <= 1:
        return ray_depth, sim, valid
    import cv2

    h, w = ray_depth.shape
    size = (w * k, h * k)
    d = ray_depth.astype(np.float32)
    d = np.where(valid, d, float(d[valid].mean()) if valid.any() else 1.0)

    kernel = np.ones((3, 3), np.uint8)
    lo = cv2.erode(d, kernel)
    hi = cv2.dilate(d, kernel)
    jump = (hi - lo) > rtol * np.maximum(d, 1e-9)
    guard = cv2.dilate(((~valid) | jump).astype(np.uint8), kernel) > 0
    if method == "cubic":
        guard = cv2.dilate(guard.astype(np.uint8), kernel) > 0
    smooth = cv2.INTER_CUBIC if method == "cubic" else cv2.INTER_LINEAR

    # Guarded pixels are resampled nearest-neighbour rather than dropped: nearest never
    # interpolates, so it cannot invent surface between the nose edge and the cheek behind
    # it, and every measured pixel survives.  Dropping them instead costs ~8% of the face.
    guard_up = cv2.resize(guard.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    valid_up = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    depth_up = np.where(
        guard_up,
        cv2.resize(d, size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(d, size, interpolation=smooth),
    )
    sim_f = sim.astype(np.float32)
    sim_up = np.where(
        guard_up,
        cv2.resize(sim_f, size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(sim_f, size, interpolation=cv2.INTER_LINEAR),
    )
    sim_up = np.clip(sim_up, -1.0, 0.0)
    depth_up[~valid_up] = INVALID_DEPTH
    sim_up[~valid_up] = 0.0
    return depth_up.astype(np.float32), sim_up.astype(np.float32), valid_up


def export_frames(
    geoms: Sequence[FrameGeometry], k: int, out_dir: Path, args: argparse.Namespace, exr: ExrWriter
) -> dict[int, Path]:
    depth_dir = out_dir / "depthMaps"
    image_dir = out_dir / "undistorted"
    depth_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths: dict[int, Path] = {}
    image_ext = args.image_format.lower().lstrip(".")
    kept_fractions = []

    for geom in geoms:
        assert geom.depth is not None and geom.conf is not None
        frame_w, frame_h = geom.frame_size

        # ---- validity + depth + similarity ---------------------------------------
        valid, ray_depth = _view_validity(geom, args)
        ray_depth = ray_depth.copy()
        ray_depth[~valid] = INVALID_DEPTH
        sim = confidence_to_similarity(geom.conf, valid)

        # scale-1 == image resolution
        K_img = scaled_intrinsic(geom.intrinsic, k, args.pixel_center == "aligned")
        if args.depth_resolution == "image":
            ray_depth, sim, valid = upsample_to_image_grid(
                ray_depth, sim, valid, k, args.depth_edge_rtol or 0.03, args.depth_upsample
            )
            depth_downscale = 1
            K_depth = K_img
        else:
            depth_downscale = int(k)
            K_depth = geom.intrinsic
        R, t = geom.extrinsic[:3, :3], geom.extrinsic[:3, 3]
        center = (-R.T @ t) * args.scene_scale
        i_cam = R.T @ np.linalg.inv(K_depth)  # at depth-map resolution
        valid_depths = ray_depth[ray_depth > 0]

        common_meta = {
            "AliceVision:downscale": int(depth_downscale),
            "AliceVision:roiBeginX": 0,
            "AliceVision:roiBeginY": 0,
            "AliceVision:roiEndX": int(frame_w * k),
            "AliceVision:roiEndY": int(frame_h * k),
            "AliceVision:tileBufferWidth": int(frame_w * k),
            "AliceVision:tileBufferHeight": int(frame_h * k),
            "AliceVision:tilePadding": 0,
            "AliceVision:P": ("matrix44", projection_matrix(K_img, geom.extrinsic).flatten().tolist()),
            "AliceVision:CArr": ("vec3", center.tolist()),
            "AliceVision:iCamArr": ("matrix33", i_cam.flatten().tolist()),
        }
        depth_meta = dict(common_meta)
        depth_meta.update(
            {
                "AliceVision:nbDepthValues": int(valid_depths.size),
                "AliceVision:minDepth": float(valid_depths.min()) if valid_depths.size else -1.0,
                "AliceVision:maxDepth": float(valid_depths.max()) if valid_depths.size else -1.0,
            }
        )

        exr.write(depth_dir / f"{geom.view_id}_depthMap.exr", ray_depth, depth_meta)
        exr.write(depth_dir / f"{geom.view_id}_simMap.exr", sim, common_meta)

        # ---- texture image ------------------------------------------------------
        image = render_at_scale(geom.texture_path or geom.path, geom, k)
        image = apply_gain(image, geom.gain)
        image_path = image_dir / f"{geom.view_id}.{image_ext}"
        if image_ext in ("jpg", "jpeg"):
            image.save(image_path, quality=args.jpeg_quality, subsampling=0)
        else:
            image.save(image_path)
        image_paths[geom.view_id] = image_path

        kept = 100.0 * valid.sum() / valid.size
        kept_fractions.append(kept)
        log(
            f"  view {geom.view_id:>4}  depth {frame_w}x{frame_h}  image {frame_w*k}x{frame_h*k}"
            f"  kept {kept:5.1f}%  gain [{geom.gain[0]:.2f} {geom.gain[1]:.2f} {geom.gain[2]:.2f}]"
            f"  {geom.path.name}"
        )

    log(f"mean kept depth pixels: {np.mean(kept_fractions):.1f}%")
    if np.mean(kept_fractions) < 5:
        log("WARNING: very few depth pixels survived filtering; loosen --conf-percentile / --depth-edge-rtol")
    return image_paths


# --------------------------------------------------------------------------------------
# 8. AliceVision driver
# --------------------------------------------------------------------------------------


class AliceVision:
    def __init__(self, bin_dir: Path | None, verbose: str = "info", dry_run: bool = False) -> None:
        self.bin_dir = Path(bin_dir) if bin_dir else None
        self.verbose = verbose
        self.dry_run = dry_run
        self._cache: dict[str, str] = {}

    def resolve(self, tool: str) -> str:
        if tool in self._cache:
            return self._cache[tool]
        candidates: list[Path] = []
        if self.bin_dir:
            candidates += [self.bin_dir / tool, self.bin_dir / f"{tool}.exe"]
        env_root = os.environ.get("ALICEVISION_ROOT")
        if env_root:
            candidates += [Path(env_root) / "bin" / tool, Path(env_root) / "bin" / f"{tool}.exe"]
        for candidate in candidates:
            if candidate.is_file():
                self._cache[tool] = str(candidate)
                return self._cache[tool]
        found = shutil.which(tool)
        if found:
            self._cache[tool] = found
            return found
        if self.dry_run:
            return tool
        raise FileNotFoundError(
            f"Cannot find '{tool}'. Pass --av-bin <AliceVision>/bin, set ALICEVISION_ROOT, "
            "or put the binaries on PATH."
        )

    def run(self, tool: str, **kwargs: Any) -> None:
        cmd = [self.resolve(tool)]
        for key, value in kwargs.items():
            if value is None:
                continue
            flag = f"--{key}"
            if isinstance(value, bool):
                cmd += [flag, "True" if value else "False"]
            elif isinstance(value, (list, tuple)):
                cmd += [flag] + [str(v) for v in value]
            else:
                cmd += [flag, str(value)]
        cmd += ["--verboseLevel", self.verbose]

        log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if self.dry_run:
            return
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"{tool} failed with exit code {result.returncode}")


# --------------------------------------------------------------------------------------
# 9. input discovery
# --------------------------------------------------------------------------------------


def collect_images(images_arg: Path) -> list[Path]:
    if images_arg.is_dir():
        paths = sorted(p for p in images_arg.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = [Path(line.strip()) for line in images_arg.read_text().splitlines() if line.strip()]
    if not paths:
        raise SystemExit(f"No images found in {images_arg}")
    return paths


def pair_auxiliary(images: Sequence[Path], folder: Path | None) -> list[Path | None]:
    """Match an auxiliary file (mask / corrected image) to each input image.

    Tries stem-prefix matching first (image_010.jpg -> image_010_dilation_mask.jpg),
    then falls back to sorted-index pairing.
    """
    if folder is None:
        return [None] * len(images)
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not candidates:
        raise SystemExit(f"No files found in auxiliary folder {folder}")

    by_stem = {p.stem: p for p in candidates}
    out: list[Path | None] = []
    matched_by_stem = 0
    for image in images:
        hit = by_stem.get(image.stem)
        if hit is None:
            prefixed = [p for p in candidates if p.stem.startswith(image.stem)]
            hit = prefixed[0] if len(prefixed) >= 1 else None
        if hit is not None:
            matched_by_stem += 1
        out.append(hit)

    if matched_by_stem < len(images):
        if len(candidates) != len(images):
            raise SystemExit(
                f"Cannot pair {folder} ({len(candidates)} files) with {len(images)} images: "
                "names do not match and the counts differ."
            )
        log(f"  {folder.name}: pairing by sorted index ({len(candidates)} files)")
        out = list(candidates)
    else:
        log(f"  {folder.name}: paired {matched_by_stem}/{len(images)} by name")
    return out


def default_output_dir(images_dir: Path, dense: bool = False) -> Path:
    """<images_dir>_alicevision (or vggt_omega_dense_textured_mesh), next to --images.

    Keeps inputs and outputs side by side (e.g. .../albert_185119/images ->
    .../albert_185119/images_alicevision) so it's obvious which images produced
    which output, without having to track a separately-chosen --output root.
    """
    images_dir = images_dir.resolve()
    if dense:
        return images_dir.parent / "vggt_omega_dense_textured_mesh"
    return images_dir.parent / f"{images_dir.name}_alicevision"


def first_existing(root: Path, relatives: Iterable[str]) -> Path | None:
    for relative in relatives:
        candidate = root / relative
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def choose_image_scale(geoms: Sequence[FrameGeometry], requested: int, max_side: int) -> int:
    """Integer factor between texture images and depth maps.

    0 = pick whatever puts the texture images closest to the input resolution.
    """
    if requested > 0:
        k = requested
    else:
        ratios = [
            (geom.crop_box[2] - geom.crop_box[0]) / max(geom.net_size[0], 1) for geom in geoms
        ]
        k = max(1, int(round(float(np.median(ratios)))))
        log(f"auto image scale k={k} (median input/network width ratio {np.median(ratios):.2f})")

    frame_w, frame_h = geoms[0].frame_size
    while k > 1 and max(frame_w * k, frame_h * k) > max_side:
        k -= 1
        log(f"  reducing k to {k} to stay under --max-image-side {max_side}")
    return k


# --------------------------------------------------------------------------------------
# 10. CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VGGT-Omega inference -> AliceVision meshing & texturing -> textured .obj",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument("--images", type=Path, help="folder of images, or a .txt list")
    io_group.add_argument("--masks", type=Path, default=None,
                          help="foreground masks (white = subject); background depth is discarded")
    io_group.add_argument("--texture-images", type=Path, default=None,
                          help="alternative, pixel-aligned images used for texturing only "
                               "(e.g. a brightness/CCT-corrected copy)")
    io_group.add_argument("--batch-root", type=Path, default=None,
                          help="process every capture under this root (see --batch-glob)")
    io_group.add_argument("--batch-glob", default="*/images",
                          help="glob, relative to --batch-root, matching each capture's image folder")
    io_group.add_argument("--auto-aux", action="store_true", default=True,
                          help="in batch mode, auto-detect bgrm masks and brightness-CCT-adjust images")
    io_group.add_argument("--no-auto-aux", action="store_false", dest="auto_aux")
    io_group.add_argument("--checkpoint", type=Path, required=True, help="VGGT-Omega .pt checkpoint")
    io_group.add_argument("--output", type=Path, default=None,
                          help="output folder. Default: <images_folder>_alicevision, created next to "
                               "the images folder itself (e.g. .../albert_185119/images -> "
                               "'.../albert_185119/images_alicevision'), so it's obvious which images "
                               "produced which output")
    io_group.add_argument("--av-bin", type=Path, default=None, dest="av_bin", help="AliceVision bin folder")

    net = parser.add_argument_group("VGGT-Omega")
    net.add_argument("--image-resolution", type=int, default=512, help="512 for the 1B-512 checkpoint")
    net.add_argument("--device", default="auto", help="auto (CUDA when available, otherwise CPU), cuda / cuda:0 / cpu")
    net.add_argument("--depth-resolution", default="image", choices=["image", "network"],
                     help="grid the depth maps are written on. 'image' resamples VGGT-Omega's "
                          "network-resolution depth onto the camera's own pixel grid, which is "
                          "what Meshroom's own DepthMapFilter hands aliceVision_meshing; "
                          "'network' writes them at inference resolution and tells AliceVision "
                          "the downscale, which costs k^2 samples AND inflates pixSize by k, "
                          "fusing away detail the data supports (see upsample_to_image_grid)")
    net.add_argument("--depth-upsample", default="linear", choices=["linear", "cubic"],
                     help="interpolation used by --depth-resolution image inside continuous regions. "
                          "'cubic' is C1 and removes the network-pixel facet grid that bilinear "
                          "prints into the mesh (v4)")
    net.add_argument("--pixel-center", default="legacy", choices=["legacy", "aligned"],
                     help="how network-resolution intrinsics are scaled by k. 'legacy' (v1-v3) "
                          "uses k*K and puts every view's rays (k-1)/2 image pixels off; "
                          "'aligned' adds the (k-1)/2 principal-point shift that centre-aligned "
                          "resizing implies (v4). See scaled_intrinsic()")

    filt = parser.add_argument_group("depth filtering")
    filt.add_argument("--conf-percentile", type=float, default=0.0,
                      help="drop the N%% least confident pixels of each view (0 = off). Measured "
                           "on patient_27_20260522_135730: 20 costs 26%% of the surface area and "
                           "leaves a hole under the chin, because VGGT-Omega's face prior is least "
                           "confident exactly where few views see the subject -- which is the part "
                           "of its output we want most. cross-view consensus already rejects "
                           "pixels no other view corroborates, and does it on evidence rather "
                           "than on the network's own self-assessment")
    filt.add_argument("--conf-min", type=float, default=0.0, help="absolute confidence floor (0 = off)")
    filt.add_argument("--depth-edge-rtol", type=float, default=0.0,
                      help="drop pixels whose 3x3 relative depth jump exceeds this (0 = off). "
                           "Redundant since upsample_to_image_grid(), which resamples those same "
                           "pixels nearest-neighbour instead of interpolating across the jump, so "
                           "the silhouette survives instead of being discarded. That guard uses "
                           "0.03 whatever this is set to")
    filt.add_argument("--scene-scale", type=float, default=1.0, help="multiply the whole scene by this factor")

    cons = parser.add_argument_group("cross-view consensus (fixes the crinkled/cracked surface)")
    cons.add_argument("--consensus-passes", type=int, default=3,
                      help="make the per-view depth maps agree with each other before meshing "
                           "(0 = off, which is what produced the 'dried mud' crack network). "
                           "This is the equivalent of Meshroom's DepthMapFilter node, which our "
                           "pipeline otherwise skips entirely")
    cons.add_argument("--consensus-tolerance", type=float, default=20.0,
                      help="a neighbour view corroborates a pixel if it agrees within this many "
                           "pixSize; raise it if too many pixels are discarded")
    cons.add_argument("--consensus-min-agree", type=int, default=2,
                      help="discard pixels corroborated by fewer neighbours "
                           "(AliceVision DepthMapFilter's minNumOfConsistentCams). With 9 wide-"
                           "baseline views the chin underside, nose flanks and ears are seen by "
                           "one or two views only; requiring 3 threw them away and left holes. "
                           "Measured: 3 -> 2 lifts kept depth pixels 24.2%% -> 28.6%%, the mesh "
                           "329k -> 403k vertices, and closes every interior hole")
    cons.add_argument("--consensus-neighbours", type=int, default=10,
                      help="how many nearest views to test against (nNearestCams)")
    cons.add_argument("--consensus-max-view-angle", type=float, default=70.0,
                      help="ignore views whose optical axis differs by more than this")
    cons.add_argument("--consensus-smooth-radius", type=int, default=2,
                      help="edge-aware box smoothing applied after the consensus passes (0 = off)")
    cons.add_argument("--consensus-smooth-tolerance", type=float, default=3.0,
                      help="only average neighbours within this many pixSize, so real edges survive")
    cons.add_argument("--consensus-sampling", default="nearest", choices=["nearest", "bilinear"],
                      help="how a neighbour view's depth is looked up at the reprojected position. "
                           "'nearest' (v3) quantises it to the neighbour's pixel grid, which on "
                           "sloped skin becomes a regular depth ripple; 'bilinear' (v4) interpolates")
    cons.add_argument("--consensus-fuse", default="median", choices=["median", "inlier-mean"],
                      help="how corroborating depths are combined. 'median' (v3) switches source "
                           "view from pixel to pixel; 'inlier-mean' (v4) averages all values within "
                           "--consensus-fuse-band of the median")
    cons.add_argument("--consensus-fuse-band", type=float, default=2.0,
                      help="inlier band around the median for --consensus-fuse inlier-mean, in pixSize")

    tex = parser.add_argument_group("texture images")
    tex.add_argument("--image-scale", type=int, default=0,
                     help="integer factor between texture images and depth maps; 0 = auto, "
                          "i.e. as close to the input resolution as AliceVision's integer "
                          "downscale constraint allows")
    tex.add_argument("--max-image-side", type=int, default=8192,
                     help="clamp the auto image scale so no texture image exceeds this")
    tex.add_argument("--image-format", default="png", choices=["png", "jpg", "tif"])
    tex.add_argument("--jpeg-quality", type=int, default=95)
    tex.add_argument("--harmonize", default="gain", choices=["gain", "luminance", "none"],
                     help="per-view color harmonisation before texturing; 'gain' = per-channel "
                          "von Kries gain in linear RGB (best for skin-tone continuity)")

    sfm = parser.add_argument_group("SfMData")
    sfm.add_argument("--sfm-version", default="1.2.6", choices=["1.2.6", "1.2.14"],
                     help="sfmDataIO version to declare. AliceVision rejects files newer than "
                          "itself, so 1.2.6 (Meshroom 2023.3+) is the safe default")

    dense = parser.add_argument_group("dense MVS (native AliceVision depth estimation)")
    dense.add_argument("--dense-mvs", action="store_true",
                       help="replace VGGT-Omega's own (network-resolution) depth maps with "
                            "aliceVision_depthMapEstimation + aliceVision_depthMapFiltering "
                            "run on the full-resolution images -- the same PatchMatch stereo "
                            "Meshroom itself uses -- seeded by sparse landmarks synthesised "
                            "from VGGT-Omega's depth+pose. Much denser mesh (comparable to a "
                            "native Meshroom reconstruction of the same photos), at the cost "
                            "of extra GPU time. Default output folder becomes "
                            "'vggt_omega_dense_textured_mesh' next to --images.")
    dense.add_argument("--dense-landmarks-per-view", type=int, default=4000,
                       help="points sampled from VGGT-Omega's depth per view; only used to "
                            "seed AliceVision's per-camera depth search range and neighbour "
                            "camera selection, not part of the final mesh")
    dense.add_argument("--dense-landmark-depth-tol", type=float, default=0.06,
                       help="relative depth tolerance when confirming a sampled point "
                            "against another view's own depth, to accept it as a shared "
                            "(multi-view) landmark between the two")
    dense.add_argument("--dense-downscale", type=int, default=2,
                       help="aliceVision_depthMapEstimation image downscale (1 = full "
                            "resolution = densest, slowest)")
    dense.add_argument("--dense-min-view-angle", type=float, default=2.0)
    dense.add_argument("--dense-max-view-angle", type=float, default=70.0)
    dense.add_argument("--dense-min-consistent-cams", type=int, default=3,
                       help="aliceVision_depthMapFiltering minNumOfConsistentCams; lower it "
                            "(e.g. 2) for captures with few, widely-spaced views")

    ba = parser.add_argument_group("pose refinement (bundle adjustment, requires --dense-mvs)")
    ba.add_argument("--ba", action="store_true",
                    help="refine VGGT-Omega's per-view poses (and intrinsics) with a COLMAP "
                         "bundle adjustment (via pycolmap) before running "
                         "aliceVision_depthMapEstimation. Uses the --dense-mvs seed landmarks "
                         "as tracks, so it needs no separate feature matching: VGGT-Omega "
                         "supplies both the initial poses and the correspondences, pycolmap "
                         "only refines them to sub-pixel reprojection error. This is what "
                         "removes the per-pixel pose jitter that otherwise shows up as a "
                         "bumpy/orange-peel surface in the dense-MVS depth maps.")
    ba.add_argument("--ba-min-inliers-per-frame", type=int, default=32,
                    help="minimum multi-view-confirmed landmarks a view must have to take "
                         "part in the bundle adjustment; pycolmap skips BA entirely if any "
                         "view falls short, so lower this for sparse/wide-baseline captures")

    mesh = parser.add_argument_group("aliceVision_meshing")
    # defaults below match template_decimation.mg (the Meshroom graph that reconstructs
    # these faces cleanly), not AliceVision's library defaults, which differ slightly
    mesh.add_argument("--max-input-points", type=int, default=500_000_000)
    mesh.add_argument("--max-points", type=int, default=10_000_000)
    mesh.add_argument("--max-points-per-voxel", type=int, default=1_000_000)
    mesh.add_argument("--min-angle-threshold", type=float, default=1.0)
    mesh.add_argument("--min-step", type=int, default=2, help="depth-map pixel stride when fusing")
    mesh.add_argument("--min-vis", type=int, default=2, help="min. number of cameras seeing a point")
    mesh.add_argument("--sim-factor", type=float, default=15.0)
    mesh.add_argument("--dense-point-cloud-ext", default="abc", choices=["abc", "ply", "sfm"])
    # fusion tolerances, all expressed in pixSize (= depth / focal).  AliceVision's defaults
    # assume depth maps that agree to well under 1 pixSize across views (Meshroom's filtered
    # maps measure 0.36).  Raise these if you keep --consensus-passes 0 or the subject is
    # still crinkly: they widen the band in which the graph cut treats two views' surfaces
    # as the same surface rather than carving a gap between them.
    mesh.add_argument("--pix-size-margin-init-coef", type=float, default=2.0)
    mesh.add_argument("--pix-size-margin-final-coef", type=float, default=4.0)
    mesh.add_argument("--n-pixel-size-behind", type=float, default=4.0,
                      help="how far behind a point the graph cut votes 'full'")
    mesh.add_argument("--vote-margin-factor", type=float, default=4.0)
    mesh.add_argument("--contribute-margin-factor", type=float, default=2.0)
    mesh.add_argument("--sim-gaussian-size-init", type=float, default=10.0)
    mesh.add_argument("--sim-gaussian-size", type=float, default=10.0)
    mesh.add_argument("--angle-factor", type=float, default=15.0)

    post = parser.add_argument_group("mesh post-processing")
    post.add_argument("--smoothing-iterations", type=int, default=5)
    post.add_argument("--filtering-iterations", type=int, default=1)
    post.add_argument("--keep-largest-only", action="store_true", default=True)
    post.add_argument("--no-keep-largest-only", action="store_false", dest="keep_largest_only")
    post.add_argument("--decimate-factor", type=float, default=0.0,
                      help="run meshDecimate with this simplification factor (0 = skip)")
    post.add_argument("--denoise-iterations", type=int, default=1,
                      help="run aliceVision_meshDenoising with this many iterations (0 = skip). "
                           "Unlike --smoothing-iterations (blanket Laplacian smoothing, which "
                           "erases pores/wrinkles along with the noise), this is a rolling-guidance "
                           "normal filter [Wang et al. 2015]: it flattens noise on smooth areas "
                           "while preserving genuine sharp features. This is the node Meshroom's "
                           "own decimation template uses, and the right knob when the surface is "
                           "noisy but you want to keep high-frequency detail")
    # Meshroom's own decimation template uses lambda=2.0/eta=1.8/nu=0.3, but that is tuned for
    # a mesh already decimated to 20% of its vertices.  Measured per-edge on an undecimated
    # pointmap mesh, those values flatten the 10-40 deg dihedral band (fine skin relief) to
    # 16-20% of its original amplitude -- i.e. they erase detail, not just noise.  These
    # gentler values keep 71%/92% of the 20-40 deg / 40 deg+ bands while still cutting the
    # 2-10 deg band (noise on flat skin) to ~25%.
    post.add_argument("--denoise-lambda", type=float, default=0.5, help="meshDenoising regularization weight")
    post.add_argument("--denoise-eta", type=float, default=1.5,
                      help="meshDenoising spatial-weight sigma, in units of average adjacent-face distance")
    post.add_argument("--denoise-mu", type=float, default=1.5, help="meshDenoising guidance-weight sigma")
    post.add_argument("--denoise-nu", type=float, default=0.1,
                      help="meshDenoising signal-weight sigma; smaller = stricter about what counts "
                           "as a feature to preserve")

    rtx = parser.add_argument_group("registered Laplacian-pyramid re-texturing (texture_pyramid.py)")
    rtx.add_argument("--retexture", default="none", choices=["none", "pyramid"],
                     help="after aliceVision_texturing, re-bake the atlas with per-band weights, "
                          "optical-flow registration of every view, per-surface-point gains and "
                          "specular down-weighting -> texturedMesh_pyramid/ (v4: pyramid)")
    import texture_pyramid

    texture_pyramid.add_options(rtx, prefix="pyr-")

    txt = parser.add_argument_group("aliceVision_texturing")
    txt.add_argument("--texture-side", type=int, default=8192, choices=[1024, 2048, 4096, 8192, 16384])
    txt.add_argument("--texture-downscale", type=int, default=1, choices=[1, 2, 4, 8])
    txt.add_argument("--unwrap-method", default="Basic", choices=["Basic", "LSCM", "ABF"])
    txt.add_argument("--fill-holes", action="store_true", default=True)
    txt.add_argument("--no-fill-holes", action="store_false", dest="fill_holes")
    txt.add_argument("--use-udim", action="store_true", default=True)
    txt.add_argument("--no-use-udim", action="store_false", dest="use_udim")
    txt.add_argument("--texture-file-type", default="png", choices=["png", "jpg", "tif", "exr"])
    txt.add_argument("--multi-band-nb-contrib", type=int, nargs=4, default=[1, 1, 10, 0],
                     help="views averaged per frequency band, highest frequency first. Meshroom's "
                          "own 1/5/10/0 averages five views in the second band; with geometry that "
                          "is accurate but not exact those five do not register, and the iris, "
                          "eyelashes and eyelid crease -- all mid-frequency -- blur. Taking the "
                          "single best view for the top two bands keeps them sharp, while the "
                          "lowest band still averages ten views so skin tone stays continuous")
    txt.add_argument("--multi-band-downscale", type=int, default=4)
    txt.add_argument("--use-score", action="store_true", default=True)
    txt.add_argument("--no-use-score", action="store_false", dest="use_score")
    txt.add_argument("--best-score-threshold", type=float, default=0.1)
    txt.add_argument("--angle-hard-threshold", type=float, default=90.0)
    txt.add_argument("--visibility-remapping-method", default="PullPush",
                     choices=["Pull", "Push", "PullPush", "MeshItself", "Basic"])
    txt.add_argument("--subdivision-target-ratio", type=float, default=0.8)
    txt.add_argument("--texture-padding", type=int, default=5)

    flow = parser.add_argument_group("pipeline control")
    flow.add_argument("--skip-inference", action="store_true", help="reuse sfm/depth maps already in --output")
    flow.add_argument("--export-only", action="store_true", help="stop after writing the AliceVision inputs")
    flow.add_argument("--dry-run", action="store_true", help="print AliceVision commands without running them")
    flow.add_argument("--continue-on-error", action="store_true", help="batch mode: keep going after a failure")
    flow.add_argument("--verbose-level", default="info",
                      choices=["fatal", "error", "warning", "info", "debug", "trace"])

    args = parser.parse_args(argv)
    if not args.images and not args.batch_root:
        parser.error("one of --images or --batch-root is required")
    if args.ba and not args.dense_mvs:
        parser.error("--ba requires --dense-mvs (it refines the poses fed into "
                      "aliceVision_depthMapEstimation, which only runs under --dense-mvs)")
    return args


# --------------------------------------------------------------------------------------
# 11. one capture
# --------------------------------------------------------------------------------------


def process_capture(
    image_paths: Sequence[Path],
    masks: Path | None,
    texture_images: Path | None,
    out: Path,
    args: argparse.Namespace,
    av: AliceVision,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sfm_path = out / "sfm.sfm"
    depth_dir = out / "depthMaps"
    image_dir = out / "undistorted"
    mesh_dir = out / "mesh"
    texture_dir = out / "texturedMesh"

    if not args.skip_inference:
        with Section(f"VGGT-Omega inference ({len(image_paths)} images)"):
            images, geoms = preprocess_images(image_paths, image_resolution=args.image_resolution)
            log(f"network input {tuple(images.shape)}")

            mask_files = pair_auxiliary(image_paths, masks)
            texture_files = pair_auxiliary(image_paths, texture_images)
            for index, geom in enumerate(geoms):
                geom.view_id = index + 1
                geom.mask_path = mask_files[index]
                geom.texture_path = texture_files[index]

            run_vggt_omega(geoms, images, checkpoint=args.checkpoint, device=args.device)

        with Section("Export SfMData + depth/sim maps + texture images"):
            k = choose_image_scale(geoms, args.image_scale, args.max_image_side)
            compute_harmonisation_gains(geoms, args.harmonize)
            exr = ExrWriter()
            if not args.dense_mvs:
                # AliceVision runs its own DepthMapFilter in --dense-mvs mode; for the
                # VGGT point-map path this is the only cross-view consistency step there is
                cross_view_consensus(geoms, args)
            landmarks = (
                build_dense_seed_landmarks(geoms, k, args, np.random.default_rng(0)) if args.dense_mvs else []
            )
            if args.ba:
                with Section("pycolmap bundle adjustment (pose refinement)"):
                    refine_poses_with_ba(geoms, k, landmarks, args)
            written = export_frames(geoms, k, out, args, exr)
            save_sfmdata(
                build_sfmdata(geoms, k, written, args.sfm_version, landmarks, args.pixel_center == "aligned"),
                sfm_path,
            )
    else:
        if not sfm_path.is_file():
            raise SystemExit(f"--skip-inference given but {sfm_path} does not exist")
        log(f"reusing existing {sfm_path}")

    if args.export_only:
        log("--export-only: stopping before AliceVision.")
        log(f"  sfmData    : {sfm_path}")
        log(f"  depth maps : {depth_dir}")
        log(f"  images     : {image_dir}")
        return

    meshing_depth_dir = depth_dir
    if args.dense_mvs:
        depthmap_raw_dir = out / "depthMapsRaw"
        depthmap_raw_dir.mkdir(parents=True, exist_ok=True)
        with Section("aliceVision_depthMapEstimation"):
            av.run(
                "aliceVision_depthMapEstimation",
                input=sfm_path,
                imagesFolder=image_dir,
                output=depthmap_raw_dir,
                downscale=args.dense_downscale,
                minViewAngle=args.dense_min_view_angle,
                maxViewAngle=args.dense_max_view_angle,
            )
        depthmap_filtered_dir = out / "depthMapsFiltered"
        depthmap_filtered_dir.mkdir(parents=True, exist_ok=True)
        with Section("aliceVision_depthMapFiltering"):
            av.run(
                "aliceVision_depthMapFiltering",
                input=sfm_path,
                depthMapsFolder=depthmap_raw_dir,
                output=depthmap_filtered_dir,
                minViewAngle=args.dense_min_view_angle,
                maxViewAngle=args.dense_max_view_angle,
                minNumOfConsistentCams=args.dense_min_consistent_cams,
            )
        meshing_depth_dir = depthmap_filtered_dir

    mesh_dir.mkdir(parents=True, exist_ok=True)
    dense_cloud = mesh_dir / f"densePointCloud.{args.dense_point_cloud_ext}"
    raw_mesh = mesh_dir / "rawMesh.obj"
    with Section("aliceVision_meshing"):
        av.run(
            "aliceVision_meshing",
            input=sfm_path,
            depthMapsFolder=meshing_depth_dir,
            output=dense_cloud,
            outputMesh=raw_mesh,
            maxInputPoints=args.max_input_points,
            maxPoints=args.max_points,
            maxPointsPerVoxel=args.max_points_per_voxel,
            minAngleThreshold=args.min_angle_threshold,
            minStep=args.min_step,
            minVis=args.min_vis,
            simFactor=args.sim_factor,
            angleFactor=args.angle_factor,
            pixSizeMarginInitCoef=args.pix_size_margin_init_coef,
            pixSizeMarginFinalCoef=args.pix_size_margin_final_coef,
            nPixelSizeBehind=args.n_pixel_size_behind,
            voteMarginFactor=args.vote_margin_factor,
            contributeMarginFactor=args.contribute_margin_factor,
            simGaussianSizeInit=args.sim_gaussian_size_init,
            simGaussianSize=args.sim_gaussian_size,
            partitioning="singleBlock",
            repartition="multiResolution",
            # the depth maps (VGGT's own, or AliceVision's dense ones in --dense-mvs)
            # already cover the full subject, so derive the bounding box from them rather
            # than from the sparse seed landmarks used only to guide depth estimation
            estimateSpaceFromSfM=False,
            addLandmarksToTheDensePointCloud=False,
            colorizeOutput=True,
        )

    filtered_mesh = mesh_dir / "filteredMesh.obj"
    with Section("aliceVision_meshFiltering"):
        av.run(
            "aliceVision_meshFiltering",
            inputMesh=raw_mesh,
            outputMesh=filtered_mesh,
            keepLargestMeshOnly=args.keep_largest_only,
            smoothingIterations=args.smoothing_iterations,
            filteringIterations=args.filtering_iterations,
        )

    mesh_for_texturing = filtered_mesh
    if args.decimate_factor > 0:
        decimated = mesh_dir / "decimatedMesh.obj"
        with Section("aliceVision_meshDecimate"):
            av.run(
                "aliceVision_meshDecimate",
                input=filtered_mesh,
                output=decimated,
                simplificationFactor=args.decimate_factor,
            )
        mesh_for_texturing = decimated

    if args.denoise_iterations > 0:
        denoised = mesh_dir / "denoisedMesh.obj"
        with Section("aliceVision_meshDenoising"):
            av.run(
                "aliceVision_meshDenoising",
                input=mesh_for_texturing,
                output=denoised,
                denoisingIterations=args.denoise_iterations,
                meshUpdateClosenessWeight=0.001,
                **{"lambda": args.denoise_lambda},
                eta=args.denoise_eta,
                mu=args.denoise_mu,
                nu=args.denoise_nu,
                meshUpdateMethod=0,
            )
        mesh_for_texturing = denoised

    texture_dir.mkdir(parents=True, exist_ok=True)
    with Section("aliceVision_texturing"):
        av.run(
            "aliceVision_texturing",
            # texturing's --input must be the *dense* point cloud (meshing's --output),
            # not the camera-only sfm.sfm: it remaps mesh vertex visibilities from the
            # landmarks' per-point observations, which only the dense cloud carries.
            input=dense_cloud,
            inputMesh=mesh_for_texturing,
            imagesFolder=image_dir,
            output=texture_dir,
            outputMeshFileType="obj",
            textureSide=args.texture_side,
            downscale=args.texture_downscale,
            unwrapMethod=args.unwrap_method,
            fillHoles=args.fill_holes,
            useUDIM=args.use_udim,
            colorMappingFileType=args.texture_file_type,
            multiBandNbContrib=args.multi_band_nb_contrib,
            multiBandDownscale=args.multi_band_downscale,
            useScore=args.use_score,
            bestScoreThreshold=args.best_score_threshold,
            angleHardThreshold=args.angle_hard_threshold,
            visibilityRemappingMethod=args.visibility_remapping_method,
            subdivisionTargetRatio=args.subdivision_target_ratio,
            padding=args.texture_padding,
            # our images carry no AliceVision:EVComp metadata, and --harmonize already
            # equalised exposure, so this would only force a pointless linear round-trip
            correctEV=False,
        )

    if args.retexture == "pyramid" and not args.dry_run:
        import av_mesh_io
        import texture_pyramid

        with Section("registered Laplacian-pyramid re-texturing"):
            cams = av_mesh_io.load_sfm_cameras(sfm_path, image_dir)
            opts = texture_pyramid.options_from_args(args, prefix="pyr-")
            texture_pyramid.retexture(texture_dir / "texturedMesh.obj", cams, out / "texturedMesh_pyramid", opts)
        log(f"re-textured   : {out / 'texturedMesh_pyramid' / 'texturedMesh.obj'}")

    log("")
    log(f"textured mesh : {texture_dir / 'texturedMesh.obj'}")
    log(f"textures      : {texture_dir}/texture_*.{args.texture_file_type}")


# --------------------------------------------------------------------------------------
# 12. main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = resolve_device(args.device, required=not (args.skip_inference or args.dry_run))
    # --output, if given, is an explicit override root (batch: out_root/<capture>,
    # single: out_root itself). Otherwise every capture gets its own
    # <images_dir>_alicevision, next to that capture's images folder.
    out_root = args.output.resolve() if args.output else None
    av = AliceVision(args.av_bin, verbose=args.verbose_level, dry_run=args.dry_run)

    if args.batch_root:
        roots = sorted(args.batch_root.glob(args.batch_glob))
        roots = [r for r in roots if r.is_dir()]
        if not roots:
            raise SystemExit(f"No captures matched {args.batch_root / args.batch_glob}")
        log(f"batch: {len(roots)} captures under {args.batch_root}")

        failures = []
        for index, images_dir in enumerate(roots, start=1):
            capture = images_dir.parent
            log("")
            log("#" * 74)
            log(f"# [{index}/{len(roots)}] {capture.name}")
            log("#" * 74)

            masks = args.masks
            texture_images = args.texture_images
            if args.auto_aux:
                masks = masks or first_existing(capture, AUTO_MASK_CANDIDATES)
                texture_images = texture_images or first_existing(capture, AUTO_TEXTURE_CANDIDATES)
            if masks:
                log(f"  masks          : {masks}")
            if texture_images:
                log(f"  texture images : {texture_images}")

            out = (out_root / capture.name) if out_root else default_output_dir(images_dir, dense=args.dense_mvs)
            log(f"  output         : {out}")

            try:
                process_capture(collect_images(images_dir), masks, texture_images, out, args, av)
            # Helpers use SystemExit for user-facing validation failures (for
            # example a missing reusable sfm.sfm).  In batch mode this must be
            # handled just like any other per-capture failure, otherwise
            # --continue-on-error has no effect and the whole batch stops at the
            # first bad capture.
            except (Exception, SystemExit) as exc:
                failures.append((capture.name, exc))
                log(f"FAILED {capture.name}: {exc}")
                traceback.print_exc()
                if not args.continue_on_error:
                    return 1

        if failures:
            log("")
            log(f"{len(failures)}/{len(roots)} captures failed:")
            for name, exc in failures:
                log(f"  - {name}: {exc}")
            return 1
        return 0

    out = out_root if out_root else default_output_dir(args.images, dense=args.dense_mvs)
    log(f"output: {out}")
    process_capture(collect_images(args.images), args.masks, args.texture_images, out, args, av)
    return 0


if __name__ == "__main__":
    sys.exit(main())
