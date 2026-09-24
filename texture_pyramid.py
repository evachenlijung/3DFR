#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-bake the texture of an AliceVision textured mesh with registered, band-limited
multi-view blending (Laplacian pyramid for colour, Gaussian pyramid for weights).

    <output>/sfm.sfm + undistorted/*.png + texturedMesh/texturedMesh.obj (+ UV atlas)
        --> <output>/texturedMesh_pyramid/texturedMesh.obj + texture_*.png

Why not just tune aliceVision_texturing?
----------------------------------------
AliceVision's multi-band blending (Texturing.cpp) is limited in three ways that
VGGT_OMEGA_TO_ALICEVISION.md §7.5-7.7 measured on our captures:

1.  View choice is per TRIANGLE and per band, with integer "number of contributing
    views" per band, cumulated with partial_sum.  Band 1 can never average fewer than 2
    views, the finest band always switches views hard at triangle edges, and there is no
    control between "1 view" and "2 views".  -> sharp eyes and seamless skin cannot both
    be had (§7.7 "one knob, two ends").
2.  The views are blended exactly where the mesh says they project.  With ~0.5-1 mm of
    geometry error, two views of the eyelid land 1-3 px apart (§7.6); any band that
    averages more than one view ghosts.
3.  Exposure / white balance and specular highlights are handled per image, not per
    surface point (§7.7: --harmonize measured 26.50% -> 26.47%, i.e. no effect).

What this does instead
----------------------
For every texel (UV atlas sample on the mesh) and every view that sees it:

*   Weights are computed in each view's own image space (foreshortening, resolution,
    distance to occlusion / silhouette edges, image border) and then, per band b, raised
    to a power p_b and smoothed with a Gaussian pyramid to the band's own scale
    (Burt & Adelson 1983).  p_b is continuous: p=8 is "almost only the best view",
    p=1 is "plain weighted average".  Default 8 6 4 2 1 1, finest band first.  The
    smoothing makes every view transition as wide as the band's wavelength, so there are
    no triangle-edge seams in any band.
*   Every view is REGISTERED to a sharp reference before blending: the reference texture
    (best-view-dominated blend) is rendered back into the view, and a dense optical flow
    (DIS) between the two gives, per pixel, where the mesh-predicted content really is in
    that photo.  Texels are then sampled at the flow-corrected position ("floating
    textures", Eisemann et al. 2008).  This is what lets the mid bands average several
    views WITHOUT ghosting the eyelid/iris/lashes.
*   Per-view, per-channel gains are estimated from the SAME surface points seen by
    several views (robust median ratio against the multi-view median), which is the
    correct version of --harmonize.
*   View-dependent specular highlights are down-weighted where a view is much brighter
    than the multi-view median at the same surface point.

Then the colour of texel x is  sum_b  sum_v  w_{v,b}(x) * L_b(I_v)(x + flow_v) / sum_v w_{v,b}(x),
where L_b(I_v) is the b-th Laplacian band of the photo.  Because the bands are exact
(sum_b L_b = I), a texel seen by one view is reproduced exactly.

Only numpy + opencv + pillow are needed (no GPU, no OpenGL).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from av_mesh_io import (
    AV_FLIP,
    Camera,
    align_mesh_to_cameras,
    ObjMesh,
    interpolate,
    linear_to_srgb8,
    load_obj,
    load_sfm_cameras,
    parse_mtl,
    push_pull_fill,
    rasterize,
    remap_points,
    save_obj,
    srgb_to_linear,
    vertex_normals,
)


def log(msg: str = "") -> None:
    print(f"[texpyr] {msg}", flush=True)


@dataclass
class Options:
    bands: int = 6
    band_sharpness: Sequence[float] = (8.0, 6.0, 4.0, 2.0, 1.0, 1.0)
    ref_sharpness: float = 8.0
    angle_power: float = 2.0
    max_angle_deg: float = 80.0
    feather_px: float = 24.0  # at full image resolution
    edge_rtol: float = 0.01  # depth jump (relative) that counts as an occlusion edge
    vis_rtol: float = 0.002  # texel-vs-zbuffer depth tolerance (relative)
    gain: bool = True
    specular: bool = True
    specular_sigma_hi: float = 0.12  # brighter than median by this fraction -> weight e^-1
    specular_sigma_lo: float = 0.40
    specular_min_band: int = 2  # highlights are low-frequency (§7.7); keep fine bands on the best view
    stats_scale: float = 0.25  # atlas / image scale for the robust statistics pass
    align: bool = True
    flow_max_px: float = 6.0  # at full image resolution
    flow_smooth_px: float = 6.0
    texture_side: int = 0  # 0 = keep the input atlas size
    chunk: int = 4_000_000
    debug_dir: Path | None = None


# --------------------------------------------------------------------------------------
# atlas
# --------------------------------------------------------------------------------------


@dataclass
class Tile:
    name: str  # output file name
    size: int
    faces: np.ndarray  # face indices of this tile
    texel_index: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # flat pixel idx
    offset: int = 0  # start of this tile in the global texel arrays


def face_local_uv(mesh: ObjMesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-face-corner UV with the UDIM tile offset removed, and the UDIM number per face."""
    uv = mesh.VT[mesh.FT]  # (f, 3, 2)
    centre = uv.mean(1)
    tile_u = np.floor(centre[:, 0]).astype(np.int64)
    tile_v = np.floor(centre[:, 1]).astype(np.int64)
    local = uv - np.stack([tile_u, tile_v], -1)[:, None, :]
    udim = 1001 + tile_u + 10 * tile_v
    return local, udim


def build_tiles(mesh: ObjMesh, mesh_path: Path, texture_side: int) -> tuple[list[Tile], np.ndarray]:
    from PIL import Image

    local_uv, udim = face_local_uv(mesh)
    mtl_map: dict[str, str] = {}
    if mesh.mtllib and (mesh_path.parent / mesh.mtllib).is_file():
        mtl_map = parse_mtl(mesh_path.parent / mesh.mtllib)

    # a tile is one texture file; faces go there by material, else by UDIM number
    keys = []
    for fi in range(len(mesh.F)):
        mat = mesh.materials[mesh.face_material[fi]] if mesh.materials else ""
        keys.append(mtl_map.get(mat) or f"texture_{int(udim[fi])}.png")
    keys = np.asarray(keys)
    tiles = []
    for name in sorted(set(keys.tolist())):
        size = texture_side
        src = mesh_path.parent / name
        if size <= 0 and src.is_file():
            with Image.open(src) as im:
                size = max(im.size)
        size = size or 4096
        tiles.append(Tile(Path(name).with_suffix(".png").name, int(size), np.nonzero(keys == name)[0]))
    return tiles, local_uv


def rasterize_atlas(tile: Tile, local_uv: np.ndarray, size: int, F_count: int):
    """(texel flat index, face, bary) for every texel centre covered by the tile's charts."""
    faces_uv = local_uv[tile.faces]  # (n, 3, 2)
    xy = np.empty((len(tile.faces) * 3, 2))
    xy[:, 0] = faces_uv[:, :, 0].reshape(-1) * size - 0.5
    xy[:, 1] = (1.0 - faces_uv[:, :, 1].reshape(-1)) * size - 0.5
    faces = np.arange(len(tile.faces) * 3).reshape(-1, 3)
    _, tri, bary = rasterize(xy, faces, size, size)
    flat = np.nonzero(tri.reshape(-1) >= 0)[0]
    face = tile.faces[tri.reshape(-1)[flat]]
    b = bary.reshape(-1, 2)[flat]
    return flat, face, b


# --------------------------------------------------------------------------------------
# per-view image-space quantities
# --------------------------------------------------------------------------------------


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pyr_blur(image: np.ndarray, levels: int) -> np.ndarray:
    """Gaussian blur at scale ~2^levels via a down/up pyramid (fast for large sigma)."""
    import cv2

    if levels <= 0:
        return image
    sizes = []
    cur = image
    for _ in range(levels):
        sizes.append((cur.shape[1], cur.shape[0]))
        if min(cur.shape[:2]) < 4:
            break
        cur = cv2.pyrDown(cur)
    for w, h in reversed(sizes[: len(sizes)]):
        cur = cv2.pyrUp(cur, dstsize=(w, h))
    return cur


@dataclass
class ViewRender:
    cam: Camera
    zbuf: np.ndarray
    tri: np.ndarray
    bary: np.ndarray
    fg: np.ndarray
    edge_dist: np.ndarray  # px to nearest occlusion/silhouette edge
    border: np.ndarray  # feather to the image border (0..1)
    base: np.ndarray  # base weight image


def render_view(cam: Camera, mesh: ObjMesh, VN: np.ndarray, opts: Options, res_ref: float, scale: float) -> ViewRender:
    import cv2

    u, v, z = cam.project(mesh.V)
    xy = np.stack([u, v], -1)
    zbuf, tri, bary = rasterize(xy, mesh.F, cam.width, cam.height, depth=z, chunk=opts.chunk)
    fg = tri >= 0
    H, W = fg.shape
    feather = opts.feather_px * scale

    zb = np.where(fg, zbuf, 0).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(zb, kernel)
    zmin = cv2.erode(np.where(fg, zb, np.float32(1e30)), kernel)
    edge = (~fg) | ((zmax - zmin) > opts.edge_rtol * np.maximum(zb, 1e-9))
    edge_dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)

    ys, xs = np.mgrid[0:H, 0:W]
    border_d = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys)).astype(np.float32)
    border = smoothstep(border_d / max(feather, 1.0))

    base = np.zeros((H, W), np.float32)
    if fg.any():
        t = tri[fg]
        b = bary[fg]
        P = interpolate(mesh.V, mesh.F, t, b)
        N = interpolate(VN, mesh.F, t, b)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        d = cam.C[None, :] - P
        dist = np.linalg.norm(d, axis=1)
        cos = np.einsum("ij,ij->i", N, d / dist[:, None])
        cos_min = np.cos(np.radians(opts.max_angle_deg))
        ang = smoothstep((cos - cos_min) / max(1.0 - cos_min, 1e-6) * 4.0) * np.clip(cos, 0, 1) ** opts.angle_power
        res = (cam.K[0, 0] / scale / dist) / res_ref  # pixel density, scale-independent
        base[fg] = (ang * res**2).astype(np.float32)
    base *= smoothstep(edge_dist / max(feather, 1.0)) * border
    return ViewRender(cam, zbuf, tri, bary, fg, edge_dist, border, base)


def band_weight_image(vr: ViewRender, p: float, band: int, opts: Options, scale: float) -> np.ndarray:
    """Weight image of one band: sharpen (power p), smooth to the band's scale, then
    force it to zero at occlusion edges over a ramp at least as wide as the band."""
    w = pyr_blur(np.power(vr.base, p, dtype=np.float32), band)
    ramp = max(opts.feather_px * scale, 3.0 * (2.0**band))
    return (w * smoothstep(vr.edge_dist / ramp) * vr.border).astype(np.float32)


def project_texels(cam: Camera, P: np.ndarray, zbuf: np.ndarray, opts: Options, N: np.ndarray | None = None):
    """(u, v, visible) for texel positions P in camera cam, occlusion-tested against zbuf."""
    u, v, z = cam.project(P.astype(np.float64))
    H, W = zbuf.shape
    inside = (z > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    vis = np.zeros(len(P), bool)
    idx = np.nonzero(inside)[0]
    iu = np.clip(np.round(u[idx]).astype(np.int64), 0, W - 1)
    iv = np.clip(np.round(v[idx]).astype(np.int64), 0, H - 1)
    zb = zbuf[iv, iu]
    ok = np.abs(z[idx] - zb) <= opts.vis_rtol * z[idx]
    if N is not None:
        d = cam.C[None, :] - P[idx]
        ok &= np.einsum("ij,ij->i", N[idx], d) > 0
    vis[idx[ok]] = True
    return u.astype(np.float32), v.astype(np.float32), vis


def load_linear_image(cam: Camera) -> np.ndarray:
    """Photo in linear RGB at the camera's resolution (area-downscaled if the camera is a
    scaled() copy of the full-resolution one)."""
    import cv2
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(cam.image_path) as im:
        arr = np.asarray(im.convert("RGB"))
    img = srgb_to_linear(arr)
    if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
        img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    return img


def laplacian_bands(img: np.ndarray, n: int) -> list[np.ndarray]:
    """Full-resolution band images whose sum is exactly img (last one = low-pass)."""
    import cv2

    gauss = [img]
    for _ in range(n - 1):
        if min(gauss[-1].shape[:2]) < 4:
            break
        gauss.append(cv2.pyrDown(gauss[-1]))

    def up_to_full(a: np.ndarray, level: int) -> np.ndarray:
        for lv in range(level, 0, -1):
            h, w = gauss[lv - 1].shape[:2]
            a = cv2.pyrUp(a, dstsize=(w, h))
        return a

    bands = []
    for lv in range(len(gauss) - 1):
        h, w = gauss[lv].shape[:2]
        lap = gauss[lv] - cv2.pyrUp(gauss[lv + 1], dstsize=(w, h))
        bands.append(up_to_full(lap, lv))
    bands.append(up_to_full(gauss[-1], len(gauss) - 1))
    while len(bands) < n:  # tiny images: pad with empty bands
        bands.insert(-1, np.zeros_like(img))
    return bands


# --------------------------------------------------------------------------------------
# robust multi-view statistics (gains + specular)
# --------------------------------------------------------------------------------------


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """values, weights: (V, N) -> (N,). Zero-weight entries are ignored."""
    order = np.argsort(values, axis=0)
    v = np.take_along_axis(values, order, 0)
    w = np.take_along_axis(weights, order, 0)
    cw = np.cumsum(w, 0)
    half = 0.5 * cw[-1]
    idx = np.argmax(cw >= half[None, :], axis=0)
    return v[idx, np.arange(values.shape[1])]


def _wmedian_1d(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


# --------------------------------------------------------------------------------------
# main algorithm
# --------------------------------------------------------------------------------------


def retexture(mesh_path: Path, cams: Sequence[Camera], out_dir: Path, opts: Options) -> dict:
    import cv2
    from PIL import Image

    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_obj(mesh_path)
    if mesh.VT is None or mesh.FT is None:
        raise SystemExit(f"{mesh_path} has no UVs; run aliceVision_texturing first (it unwraps the mesh)")
    mesh.V, flipped = align_mesh_to_cameras(mesh.V, cams)
    VN = vertex_normals(mesh.V, mesh.F)
    log(f"mesh {mesh_path.name}: {len(mesh.V):,} vertices, {len(mesh.F):,} faces, {len(cams)} views"
        + (" (AliceVision y/z-flipped frame -> camera frame)" if flipped else ""))

    # resolution reference so weights live in ~[0, 1]
    dists = [np.median(np.linalg.norm(mesh.V - c.C[None, :], axis=1)) for c in cams]
    res_ref = float(np.median([c.K[0, 0] / d for c, d in zip(cams, dists)]))

    tiles, local_uv = build_tiles(mesh, mesh_path, opts.texture_side)

    # ---- texel sets: full resolution and statistics resolution -----------------------
    def texel_set(scale: float):
        P_all, N_all, info = [], [], []
        offset = 0
        for tile in tiles:
            size = max(8, int(round(tile.size * scale)))
            flat, face, b = rasterize_atlas(tile, local_uv, size, len(mesh.F))
            P_all.append(interpolate(mesh.V, mesh.F, face, b).astype(np.float32))
            Nn = interpolate(VN, mesh.F, face, b)
            N_all.append((Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)).astype(np.float32))
            info.append((size, flat, offset))
            offset += len(flat)
        return np.concatenate(P_all), np.concatenate(N_all), info

    P, N, tex_info = texel_set(1.0)
    log(f"atlas: {len(tiles)} tile(s) {[t.size for t in tiles]}, {len(P):,} texels")

    nb = opts.bands
    sharp = list(opts.band_sharpness) + [opts.band_sharpness[-1]] * max(0, nb - len(opts.band_sharpness))
    sharp = sharp[:nb]
    report: dict = {"views": [c.view_id for c in cams], "band_sharpness": sharp}

    # ---- pass 0: robust per-surface-point statistics at low resolution -----------------
    gains = np.ones((len(cams), 3), np.float32)
    spec_tiles: list[list[np.ndarray]] | None = None
    if opts.gain or opts.specular:
        s = opts.stats_scale
        Pl, Nl, info_l = texel_set(s)
        C = np.zeros((len(cams), len(Pl), 3), np.float32)
        Wl = np.zeros((len(cams), len(Pl)), np.float32)
        for vi, cam in enumerate(cams):
            cs = cam.scaled(s)
            vr = render_view(cs, mesh, VN, opts, res_ref, s)
            img = cv2.GaussianBlur(load_linear_image(cs), (0, 0), 1.0)
            u, v, vis = project_texels(cs, Pl, vr.zbuf, opts, Nl)
            idx = np.nonzero(vis)[0]
            C[vi, idx] = remap_points(img, u[idx], v[idx])
            Wl[vi, idx] = remap_points(vr.base, u[idx], v[idx]) + 1e-6
        count = (Wl > 0).sum(0)
        multi = count >= 3
        M = np.stack([weighted_median(C[:, multi, c], Wl[:, multi]) for c in range(3)], -1)
        if opts.gain:
            Y = C[:, multi].mean(-1)
            Ym = M.mean(-1)
            dark = 0.1 * float(np.median(Ym)) if Ym.size else 0.0  # relative: exposure varies per rig
            for vi in range(len(cams)):
                # compare against the OTHER views: with <10 views the plain median is often
                # this view's own value, which pins the ratio at exactly 1
                others = [j for j in range(len(cams)) if j != vi]
                Wo = Wl[others][:, multi]
                Mo = np.stack([weighted_median(C[others][:, multi, c], Wo) for c in range(3)], -1)
                w = Wl[vi, multi]
                r = Y[vi] / np.maximum(Mo.mean(-1), 1e-6)
                sel = np.nonzero((w > 0) & ((Wo > 0).sum(0) >= 2) & (Ym > dark) & (r > 0.7) & (r < 1.4))[0]
                if sel.size > 200:  # r bounds skip highlights and shadows
                    ratio = C[vi, multi][sel] / np.maximum(Mo[sel], 1e-6)
                    gains[vi] = [_wmedian_1d(ratio[:, c], w[sel]) for c in range(3)]
            gains /= np.exp(np.log(np.maximum(gains, 1e-6)).mean(0, keepdims=True))
            C /= gains[:, None, :]
            M = np.stack([weighted_median(C[:, multi, c], Wl[:, multi]) for c in range(3)], -1)
            log("per-view gains (R G B): " + "  ".join(
                f"{c.view_id}:{g[0]:.2f}/{g[1]:.2f}/{g[2]:.2f}" for c, g in zip(cams, gains)))
        report["gains"] = gains.tolist()
        if opts.specular:
            Ym = np.zeros(len(Pl), np.float32)
            Ym[multi] = M.mean(-1)
            spec_tiles = []
            for vi in range(len(cams)):
                r = C[vi].mean(-1) / np.maximum(Ym, 1e-6)
                f = np.exp(-np.square(np.maximum(r - 1.0, 0) / opts.specular_sigma_hi)
                           - np.square(np.maximum(1.0 - r, 0) / opts.specular_sigma_lo)).astype(np.float32)
                known = multi & (Wl[vi] > 0)
                f[~known] = 1.0
                per_tile = []
                for size, flat, off in info_l:
                    img = np.ones(size * size, np.float32)
                    m = np.zeros(size * size, bool)
                    img[flat] = f[off : off + len(flat)]
                    m[flat] = known[off : off + len(flat)]
                    filled = push_pull_fill(img.reshape(size, size), m.reshape(size, size))
                    per_tile.append(cv2.GaussianBlur(filled, (0, 0), 1.0))
                spec_tiles.append(per_tile)
            log("specular down-weighting: done")
        del C, Wl, Pl, Nl

    def spec_factor(vi: int) -> np.ndarray:
        if spec_tiles is None:
            return np.ones(len(P), np.float32)
        out = np.ones(len(P), np.float32)
        for (size, flat, off), (size_l, _, _), img in zip(tex_info, info_l_sizes, spec_tiles[vi]):
            x = (flat % size + 0.5) * (size_l / size) - 0.5
            y = (flat // size + 0.5) * (size_l / size) - 0.5
            out[off : off + len(flat)] = remap_points(img, x.astype(np.float32), y.astype(np.float32))
        return out

    info_l_sizes = [(max(8, int(round(t.size * opts.stats_scale))), None, None) for t in tiles]

    # ---- pass 1: reference texture + per-band denominators ------------------------------
    tiny = np.float32(1e-20)
    ref_num = np.zeros((len(P), 3), np.float32)
    ref_den = np.zeros(len(P), np.float32)
    den = np.zeros((nb, len(P)), np.float32)
    nviews = np.zeros(len(P), np.uint8)
    for vi, cam in enumerate(cams):
        vr = render_view(cam, mesh, VN, opts, res_ref, 1.0)
        img = load_linear_image(cam) / gains[vi][None, None, :]
        u, v, vis = project_texels(cam, P, vr.zbuf, opts, N)
        idx = np.nonzero(vis)[0]
        uu, vv = u[idx], v[idx]
        sf = spec_factor(vi)[idx]
        wref = remap_points(band_weight_image(vr, opts.ref_sharpness, 1, opts, 1.0), uu, vv) * sf + tiny
        ref_num[idx] += wref[:, None] * remap_points(img, uu, vv)
        ref_den[idx] += wref
        for b in range(nb):
            sfb = sf if b >= opts.specular_min_band else 1.0
            den[b, idx] += remap_points(band_weight_image(vr, sharp[b], b, opts, 1.0), uu, vv) * sfb + tiny
        nviews[idx] += 1
        log(f"  pass 1 view {cam.view_id}: {len(idx):,} texels visible")
    seen = ref_den > 0
    ref = np.where(seen[:, None], ref_num / np.maximum(ref_den, tiny)[:, None], 0).astype(np.float32)
    del ref_num
    report["texels"] = int(len(P))
    report["texels_seen"] = int(seen.sum())
    report["mean_views_per_texel"] = float(nviews[seen].mean()) if seen.any() else 0.0

    def to_tile_images(values: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        images = []
        for (size, flat, off) in tex_info:
            img = np.zeros((size * size, values.shape[1]), np.float32)
            m = np.zeros(size * size, bool)
            img[flat] = values[off : off + len(flat)]
            m[flat] = mask[off : off + len(flat)]
            images.append(push_pull_fill(img.reshape(size, size, -1), m.reshape(size, size)))
        return images

    ref_tiles = to_tile_images(ref, seen) if opts.align else None

    # ---- pass 2: registered band-wise blend ---------------------------------------------
    acc = np.zeros((len(P), 3), np.float32)
    face_tile = np.zeros(len(mesh.F), np.int32)
    for ti, tile in enumerate(tiles):
        face_tile[tile.faces] = ti
    flow_stats = {}
    for vi, cam in enumerate(cams):
        vr = render_view(cam, mesh, VN, opts, res_ref, 1.0)
        img = load_linear_image(cam) / gains[vi][None, None, :]
        u, v, vis = project_texels(cam, P, vr.zbuf, opts, N)
        idx = np.nonzero(vis)[0]
        uu, vv = u[idx], v[idx]
        if opts.align and ref_tiles is not None:
            flow = view_flow(vr, img, ref_tiles, tex_info, local_uv, face_tile, mesh, opts)
            fx = remap_points(flow[..., 0], uu, vv)
            fy = remap_points(flow[..., 1], uu, vv)
            mag = np.hypot(fx, fy)
            flow_stats[cam.view_id] = {"median_px": float(np.median(mag)) if mag.size else 0.0,
                                       "p95_px": float(np.percentile(mag, 95)) if mag.size else 0.0}
            su, sv = uu + fx, vv + fy
            if opts.debug_dir is not None:
                Path(opts.debug_dir).mkdir(parents=True, exist_ok=True)
                m = np.clip(np.hypot(flow[..., 0], flow[..., 1]) / max(opts.flow_max_px, 1e-6) * 255, 0, 255)
                cv2.imwrite(str(Path(opts.debug_dir) / f"flow_{cam.view_id}.png"), m.astype(np.uint8))
        else:
            su, sv = uu, vv
        sf = spec_factor(vi)[idx]
        bands = laplacian_bands(img, nb)
        for b in range(nb):
            sfb = sf if b >= opts.specular_min_band else 1.0
            w = remap_points(band_weight_image(vr, sharp[b], b, opts, 1.0), uu, vv) * sfb + tiny
            w /= np.maximum(den[b, idx], tiny)
            acc[idx] += w[:, None] * remap_points(bands[b], su, sv)
        log(f"  pass 2 view {cam.view_id}" + (f": flow median {flow_stats[cam.view_id]['median_px']:.2f}px, "
                                                 f"p95 {flow_stats[cam.view_id]['p95_px']:.2f}px"
                                                 if cam.view_id in flow_stats else ""))
    report["flow"] = flow_stats

    # ---- write --------------------------------------------------------------------------
    out_tiles = to_tile_images(np.clip(acc, 0, None), seen)
    for tile, img in zip(tiles, out_tiles):
        Image.fromarray(linear_to_srgb8(img)).save(out_dir / tile.name)
    count_img = to_tile_images(nviews[:, None].astype(np.float32), np.ones(len(P), bool))
    if opts.debug_dir is not None:
        Path(opts.debug_dir).mkdir(parents=True, exist_ok=True)
        for tile, img in zip(tiles, count_img):
            cv2.imwrite(str(Path(opts.debug_dir) / f"views_{tile.name}"), np.clip(img[..., 0] * 25, 0, 255).astype(np.uint8))

    # OBJ + MTL pointing at the new textures
    obj_out = out_dir / mesh_path.name
    mtl_name = mesh.mtllib or (mesh_path.stem + ".mtl")
    if mesh.mtllib and (mesh_path.parent / mesh.mtllib).is_file():
        shutil.copyfile(mesh_path, obj_out)
        mtl_map = parse_mtl(mesh_path.parent / mesh.mtllib)
        text = (mesh_path.parent / mesh.mtllib).read_text(encoding="utf-8", errors="replace")
        for mat, tex in mtl_map.items():
            text = text.replace(tex, Path(tex).with_suffix(".png").name)
        (out_dir / mesh.mtllib).write_text(text, encoding="utf-8")
    else:
        mesh_out = ObjMesh(mesh.V @ AV_FLIP if flipped else mesh.V, mesh.F, mesh.VT, mesh.FT, np.zeros(len(mesh.F), np.int32), ["material0"], mtl_name)
        save_obj(obj_out, mesh_out)
        (out_dir / mtl_name).write_text(f"newmtl material0\nKd 1 1 1\nmap_Kd {tiles[0].name}\n", encoding="utf-8")
    report["seconds"] = time.time() - t0
    (out_dir / "texture_pyramid_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"wrote {obj_out} ({report['seconds']:.0f}s)")
    return report


def view_flow(vr: ViewRender, img: np.ndarray, ref_tiles, tex_info, local_uv, face_tile, mesh: ObjMesh,
              opts: Options) -> np.ndarray:
    """Dense flow f such that the reference texture rendered into this view, Ref(p),
    matches the photo at p + f(p).  Confidence-weighted smoothing + magnitude clamp keep
    it a smooth mis-registration field rather than a texture-copying warp."""
    import cv2

    H, W = vr.fg.shape
    ref_img = img.copy()
    fg_idx = np.nonzero(vr.fg.reshape(-1))[0]
    t = vr.tri.reshape(-1)[fg_idx]
    b = vr.bary.reshape(-1, 2)[fg_idx].astype(np.float64)
    uv = (1 - b[:, :1] - b[:, 1:]) * local_uv[t, 0] + b[:, :1] * local_uv[t, 1] + b[:, 1:] * local_uv[t, 2]
    flat = ref_img.reshape(-1, 3)
    for ti, (size, _, _) in enumerate(tex_info):
        sel = face_tile[t] == ti
        if sel.any():
            x = (uv[sel, 0] * size - 0.5).astype(np.float32)
            y = ((1.0 - uv[sel, 1]) * size - 0.5).astype(np.float32)
            flat[fg_idx[sel]] = remap_points(ref_tiles[ti], x, y)

    def gray8(a):
        return cv2.cvtColor(linear_to_srgb8(a), cv2.COLOR_RGB2GRAY)

    g_ref, g_img = gray8(ref_img), gray8(img)
    # equalise local contrast between the two so the flow tracks structure, not shading
    g_ref = cv2.createCLAHE(2.0, (8, 8)).apply(g_ref)
    g_img = cv2.createCLAHE(2.0, (8, 8)).apply(g_img)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(g_ref, g_img, None)

    gx = cv2.Sobel(g_ref.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g_ref.astype(np.float32), cv2.CV_32F, 0, 1)
    conf = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 2.0)
    inner = smoothstep(vr.edge_dist / max(opts.feather_px, 1.0)) * vr.border
    conf *= inner
    sigma = max(opts.flow_smooth_px, 0.5)
    norm = cv2.GaussianBlur(conf, (0, 0), sigma) + 1e-6
    fl = np.stack([cv2.GaussianBlur(flow[..., c] * conf, (0, 0), sigma) / norm for c in range(2)], -1)
    mag = np.hypot(fl[..., 0], fl[..., 1])
    fl *= (np.minimum(mag, opts.flow_max_px) / np.maximum(mag, 1e-6))[..., None]
    fl *= inner[..., None]
    return fl.astype(np.float32)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_options(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    d = Options()
    p = prefix
    parser.add_argument(f"--{p}bands", type=int, default=d.bands,
                        help="Laplacian bands (finest first); the last one is the low-pass residual")
    parser.add_argument(f"--{p}band-sharpness", type=float, nargs="+", default=list(d.band_sharpness),
                        help="per-band weight exponent: large = best view dominates (sharp eyes), "
                             "1 = plain weighted average (seamless skin tone)")
    parser.add_argument(f"--{p}no-align", action="store_true", help="skip optical-flow registration")
    parser.add_argument(f"--{p}no-gain", action="store_true", help="skip per-surface-point gain estimation")
    parser.add_argument(f"--{p}no-specular", action="store_true", help="skip specular down-weighting")
    parser.add_argument(f"--{p}specular-min-band", type=int, default=d.specular_min_band,
                        help="first band the specular down-weighting applies to. 2 (default) keeps "
                             "the two finest bands on the geometrically best view (crisper skin); "
                             "0 applies it everywhere (most faithful to the photos, softer)")
    parser.add_argument(f"--{p}flow-max-px", type=float, default=d.flow_max_px)
    parser.add_argument(f"--{p}flow-smooth-px", type=float, default=d.flow_smooth_px)
    parser.add_argument(f"--{p}feather-px", type=float, default=d.feather_px)
    parser.add_argument(f"--{p}texture-side", type=int, default=d.texture_side,
                        help="output atlas size (0 = same as the input textures)")


def options_from_args(args: argparse.Namespace, prefix: str = "") -> Options:
    g = lambda name: getattr(args, (prefix + name).replace("-", "_"))
    return Options(
        bands=g("bands"),
        band_sharpness=tuple(g("band-sharpness")),
        align=not g("no-align"),
        gain=not g("no-gain"),
        specular=not g("no-specular"),
        specular_min_band=g("specular-min-band"),
        flow_max_px=g("flow-max-px"),
        flow_smooth_px=g("flow-smooth-px"),
        feather_px=g("feather-px"),
        texture_side=g("texture-side"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="vggt_omega_to_alicevision output folder (holds sfm.sfm, undistorted/, texturedMesh/)")
    parser.add_argument("--mesh", type=Path, default=None, help="default: <input>/texturedMesh/texturedMesh.obj")
    parser.add_argument("--sfm", type=Path, default=None, help="default: <input>/sfm.sfm")
    parser.add_argument("--images", type=Path, default=None, help="default: <input>/undistorted")
    parser.add_argument("--output", type=Path, default=None, help="default: <input>/texturedMesh_pyramid")
    parser.add_argument("--debug", action="store_true", help="write flow / view-count images")
    add_options(parser)
    args = parser.parse_args(argv)

    mesh = args.mesh or args.input / "texturedMesh" / "texturedMesh.obj"
    cams = load_sfm_cameras(args.sfm or args.input / "sfm.sfm", args.images or args.input / "undistorted")
    out = args.output or args.input / "texturedMesh_pyramid"
    opts = options_from_args(args)
    if args.debug:
        opts.debug_dir = out / "debug"
    retexture(mesh, cams, out, opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
