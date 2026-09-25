#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the VGGT-Omega -> AliceVision pipeline on an OPEN face dataset (Multiface,
Meta, CC-BY-NC 4.0) with the same camera layout as our capture rig:

    7 eye-level views spread over roughly -70..+75 degrees azimuth
    2 views from under the chin, tilted up (~45 degrees elevation)

No private data is involved; nothing from the dataset is written into the repository.

What is simulated and what is real
----------------------------------
real      photos (2048x1334), camera calibration, the subject's skin / eyes / lighting,
          per-camera colour differences, AliceVision 2023.3 meshing + texturing binaries,
          this repository's vggt_omega_to_alicevision.py export code (consensus,
          upsampling, EXR/SfM writing) and its v3 parameters.
simulated VGGT-Omega's depth maps.  They are rendered from Multiface's tracked mesh at
          VGGT's network resolution (k=3, like the real 1024 runs) and perturbed with
          per-view low-frequency offsets + pixel noise so the views disagree with each
          other the way VGGT's do.  The tracked mesh itself is a smooth ~7k-vertex fit,
          so it differs from the true face by ~1 mm -- the same order as our geometry
          error (§7.6), which is exactly what makes multi-view texturing hard.

Held-out cameras (never used for texturing) are the ground truth: the textured mesh is
rendered into them and compared with the real photo.

Usage (after downloading one Multiface frame, see --help):
    python experiments/multiface_eval.py --multiface /tmp/mf/m--20180227--0000--6795937--GHS \
        --av-root /opt/Meshroom-2023.3.0/aliceVision --work /tmp/mf_eval
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import av_mesh_io as io  # noqa: E402
import mesh_smooth  # noqa: E402
import texture_pyramid as tp  # noqa: E402

# camera layout matching the capture rig (see the header)
EYE_LEVEL = ["400053", "400042", "400012", "400016", "400013", "400028", "400059"]
UNDER_CHIN = ["400031", "400061"]
HELD_OUT = ["400030", "400048", "400060", "400004", "400039", "400017", "400018", "400029", "400069", "400051"]


def log(msg: str) -> None:
    print(f"[mf-eval] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------


def load_krt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lines = path.read_text().split("\n")
    cams, i = {}, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        name = lines[i].strip()
        K = np.array([list(map(float, lines[i + j].split())) for j in (1, 2, 3)])
        RT = np.array([list(map(float, lines[i + j].split())) for j in (5, 6, 7)])
        cams[name] = (K, RT)
        i += 9
    return cams


def multiface_camera(name: str, krt, image: Path) -> io.Camera:
    from PIL import Image

    K, RT = krt[name]
    with Image.open(image) as im:
        W, H = im.size
    return io.Camera(int(name), K.copy(), RT[:, :3].copy(), RT[:, 3].copy(), W, H, image)


def subdivide(mesh: io.ObjMesh) -> io.ObjMesh:
    """1-to-4 midpoint subdivision carrying the UVs along."""
    def split(V, F):
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        und = np.sort(e, 1)
        uniq, inv = np.unique(und, axis=0, return_inverse=True)
        mid = len(V) + inv.reshape(3, -1).T  # (f, 3): edges 01, 12, 20
        Vn = np.concatenate([V, 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])])
        a, b, c = F.T
        ab, bc, ca = mid.T
        Fn = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                             np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
        return Vn, Fn

    V, F = split(mesh.V, mesh.F)
    VT, FT = split(mesh.VT, mesh.FT)
    return io.ObjMesh(V, F, VT, FT, np.zeros(len(F), np.int32), ["material0"], None)


# --------------------------------------------------------------------------------------
# simulated VGGT output
# --------------------------------------------------------------------------------------


def vggt_like_depth(cam_net: io.Camera, mesh: io.ObjMesh, VN, rng, lf_mm: float, hf_mm: float):
    """z-depth + confidence at network resolution, with view-specific error."""
    import cv2

    u, v, z = cam_net.project(mesh.V)
    zbuf, tri, bary = io.rasterize(np.stack([u, v], -1), mesh.F, cam_net.width, cam_net.height, depth=z)
    fg = tri >= 0
    depth = np.where(fg, zbuf, 0).astype(np.float32)
    H, W = depth.shape
    # low-frequency, view-specific offset (what makes VGGT's views disagree) ...
    field = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.08 * W)
    field *= lf_mm / max(field.std(), 1e-9)
    # ... plus per-pixel noise, slightly correlated like a network's output
    noise = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7)
    noise *= hf_mm / max(noise.std(), 1e-9)
    depth = np.where(fg, depth + field + noise, 0)
    conf = np.ones_like(depth)
    if fg.any():
        P = io.interpolate(mesh.V, mesh.F, tri[fg], bary[fg])
        N = io.interpolate(VN, mesh.F, tri[fg], bary[fg])
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        d = cam_net.C[None] - P
        cos = np.einsum("ij,ij->i", N, d / np.linalg.norm(d, axis=1, keepdims=True))
        conf[fg] = 1.0 + 4.0 * np.clip(cos, 0, 1)
    return depth, conf, fg


def export_like_pipeline(v2a, train_cams, mesh, VN, work: Path, args_list: list[str], rng, lf_mm, hf_mm):
    """Run vggt_omega_to_alicevision's own export code on simulated VGGT output."""
    from PIL import Image

    img_dir = work / "photos"
    mask_dir = work / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    geoms = []
    for i, cam in enumerate(train_cams):
        name = f"{i:02d}_{cam.view_id}.png"
        if not (img_dir / name).exists():
            shutil.copyfile(cam.image_path, img_dir / name)
        W, H = cam.width, cam.height
        net_h, net_w = v2a._target_shape(H / W, args_list_value(args_list, "--image-resolution", 512), 16, "balanced")
        sx, sy = net_w / W, net_h / H
        S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]])
        K_net = S @ cam.K
        cam_net = io.Camera(cam.view_id, K_net, cam.R, cam.t, net_w, net_h)
        depth, conf, fg = vggt_like_depth(cam_net, mesh, VN, rng, lf_mm, hf_mm)
        # subject mask at full resolution (the rig has bgrm masks)
        u, v, z = cam.project(mesh.V)
        _, tri, _ = io.rasterize(np.stack([u, v], -1), mesh.F, W, H, depth=z)
        Image.fromarray(((tri >= 0) * 255).astype(np.uint8)).save(mask_dir / name)
        g = v2a.FrameGeometry(path=img_dir / name, original_size=(W, H), crop_box=(0, 0, W, H),
                              net_size=(net_w, net_h), pad=(0, 0, 0, 0), frame_size=(net_w, net_h))
        g.view_id = i + 1
        g.mask_path = mask_dir / name
        g.intrinsic = K_net
        g.extrinsic = np.concatenate([cam.R, cam.t[:, None]], 1)
        g.depth = depth
        g.conf = conf
        geoms.append(g)
    return geoms


def args_list_value(args_list, flag, default):
    return int(args_list[args_list.index(flag) + 1]) if flag in args_list else default


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------


def render_textured(mesh_path: Path, cam: io.Camera):
    """Render a textured OBJ (UDIM aware) into a camera: sRGB image + validity mask."""
    from PIL import Image

    mesh = io.load_obj(mesh_path)
    tiles, local_uv = tp.build_tiles(mesh, mesh_path, 0)
    textures = []
    for t in tiles:
        src = mesh_path.parent / t.name
        if not src.exists():  # AliceVision names, e.g. texture_1001.png
            cands = list(mesh_path.parent.glob(Path(t.name).stem + ".*"))
            src = cands[0]
        textures.append(np.asarray(Image.open(src).convert("RGB")).astype(np.float32))
    face_tile = np.zeros(len(mesh.F), np.int32)
    for i, t in enumerate(tiles):
        face_tile[t.faces] = i
    V, _ = io.align_mesh_to_cameras(mesh.V, [cam])
    u, v, z = cam.project(V)
    _, tri, bary = io.rasterize(np.stack([u, v], -1), mesh.F, cam.width, cam.height, depth=z)
    fg = tri >= 0
    out = np.zeros((cam.height, cam.width, 3), np.float32)
    idx = np.nonzero(fg.reshape(-1))[0]
    t = tri.reshape(-1)[idx]
    b = bary.reshape(-1, 2)[idx].astype(np.float64)
    uv = (1 - b[:, :1] - b[:, 1:]) * local_uv[t, 0] + b[:, :1] * local_uv[t, 1] + b[:, 1:] * local_uv[t, 2]
    flat = out.reshape(-1, 3)
    for ti, tex in enumerate(textures):
        sel = face_tile[t] == ti
        size = tex.shape[0]
        flat[idx[sel]] = io.remap_points(tex, (uv[sel, 0] * tex.shape[1] - 0.5).astype(np.float32),
                                         ((1 - uv[sel, 1]) * size - 0.5).astype(np.float32))
    return out, fg


def ssim_gray(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    import cv2

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cab = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(s[mask].mean())


def compare(render: np.ndarray, photo: np.ndarray, mask: np.ndarray) -> dict:
    """PSNR / SSIM after a per-channel gain fit (held-out camera has its own exposure),
    plus detail metrics on a difference-of-Gaussians band (fine skin texture, lashes)."""
    import cv2

    lin_r = io.srgb_to_linear(render)
    lin_p = io.srgb_to_linear(photo)
    g = np.array([np.sum(lin_r[..., c][mask] * lin_p[..., c][mask]) / max(np.sum(lin_r[..., c][mask] ** 2), 1e-9)
                  for c in range(3)])
    r = io.linear_to_srgb8(lin_r * g).astype(np.float32)
    p = photo.astype(np.float32)
    mse = float(np.mean((r[mask] - p[mask]) ** 2))
    gr = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    gp = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)

    def dog(x, s1, s2):
        return cv2.GaussianBlur(x, (0, 0), s1) - cv2.GaussianBlur(x, (0, 0), s2)

    out = {"psnr": 10 * np.log10(255.0**2 / max(mse, 1e-9)), "ssim": ssim_gray(gr, gp, mask)}
    for name, (s1, s2) in {"fine": (0.6, 1.5), "mid": (1.5, 4.0)}.items():
        dr, dp = dog(gr, s1, s2)[mask], dog(gp, s1, s2)[mask]
        out[f"{name}_energy_ratio"] = float(dr.std() / max(dp.std(), 1e-9))  # 1 = as detailed as the photo
        out[f"{name}_corr"] = float(np.corrcoef(dr, dp)[0, 1])  # 1 = same detail, same place

    # The held-out render is itself mis-registered by the mesh error (1-3 px), which hides
    # texture quality behind geometry error.  "_reg" metrics first warp the render onto the
    # photo with a smooth optical flow, so what remains is blur / ghosting / seams.
    u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(u8(gp), u8(gr), None)
    flow = np.stack([cv2.GaussianBlur(flow[..., c], (0, 0), 4.0) for c in range(2)], -1)
    H, W = gp.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    rw = cv2.remap(r, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR)
    grw = cv2.cvtColor(rw, cv2.COLOR_RGB2GRAY)
    out["psnr_reg"] = 10 * np.log10(255.0**2 / max(float(np.mean((rw[mask] - p[mask]) ** 2)), 1e-9))
    out["ssim_reg"] = ssim_gray(grw, gp, mask)
    for name, (s1, s2) in {"fine": (0.6, 1.5), "mid": (1.5, 4.0)}.items():
        dr, dp = dog(grw, s1, s2)[mask], dog(gp, s1, s2)[mask]
        out[f"{name}_corr_reg"] = float(np.corrcoef(dr, dp)[0, 1])
    return out


def eye_mask(cam: io.Camera, mesh: io.ObjMesh, fg: np.ndarray, eye_vertices: np.ndarray) -> np.ndarray:
    """Pixels within ~12 mm of the eye centres (from mesh vertices), inside the render."""
    H, W = fg.shape
    u, v, z = cam.project(eye_vertices)
    mm_px = cam.K[0, 0] / np.median(z)
    ys, xs = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W), bool)
    for uu, vv in zip(u, v):
        m |= (xs - uu) ** 2 + ((ys - vv) / 0.6) ** 2 < (14 * mm_px) ** 2
    return m & fg


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--multiface", type=Path, required=True, help="m--20180227--0000--6795937--GHS folder")
    ap.add_argument("--expression", default="E001_Neutral_Eyes_Open")
    ap.add_argument("--frame", default="000102")
    ap.add_argument("--av-root", type=Path, required=True, help="<Meshroom>/aliceVision")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--lf-mm", type=float, default=0.6, help="std of each view's low-frequency depth error")
    ap.add_argument("--hf-mm", type=float, default=0.15, help="std of per-pixel depth noise")
    ap.add_argument("--image-resolution", type=int, default=512,
                    help="VGGT token budget; 512 gives k=3 on 1334-px-wide Multiface photos")
    ap.add_argument("--pipeline-args", default="", help="extra vggt_omega_to_alicevision.py flags")
    ap.add_argument("--texture-side", type=int, default=4096)
    ap.add_argument("--skip-pipeline", action="store_true", help="reuse <work>/pipeline")
    ap.add_argument("--variants", default="all", help="comma list of retexture variants to run")
    ap.add_argument("--reuse", action="store_true", help="only re-score variants already baked")
    args = ap.parse_args()

    os.environ["ALICEVISION_ROOT"] = str(args.av_root)
    os.environ["LD_LIBRARY_PATH"] = f"{args.av_root}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    import vggt_omega_to_alicevision as v2a

    root = args.multiface
    krt = load_krt(root / "KRT")
    img = lambda n: root / "images" / args.expression / n / f"{args.frame}.png"
    train = [multiface_camera(n, krt, img(n)) for n in EYE_LEVEL + UNDER_CHIN]
    held = [multiface_camera(n, krt, img(n)) for n in HELD_OUT if img(n).exists()]
    gt = io.load_obj(root / "tracked_mesh" / args.expression / f"{args.frame}.obj")
    gt = subdivide(subdivide(gt))
    gt.V = mesh_smooth.taubin_smooth(gt.V, gt.F, iterations=10, normal_only=False)
    VN = io.vertex_normals(gt.V, gt.F)
    log(f"reference surface: {len(gt.V):,} vertices; train {[c.view_id for c in train]}; "
        f"held-out {[c.view_id for c in held]}")

    work = args.work
    pipe = work / "pipeline"
    pipeline_argv = ["--images", str(work / "photos"), "--checkpoint", "unused.pt",
                     "--av-bin", str(args.av_root / "bin"), "--output", str(pipe),
                     "--image-resolution", str(args.image_resolution),
                     "--texture-side", str(args.texture_side), "--skip-inference",
                     # the texture comparison is run on the v3 geometry; later flags override
                     "--depth-upsample", "linear", "--pixel-center", "legacy",
                     "--consensus-sampling", "nearest", "--consensus-fuse", "median",
                     "--retexture", "none"] + args.pipeline_args.split()
    pargs = v2a.parse_args(pipeline_argv)
    pargs.device = "cpu"
    if not args.skip_pipeline:
        rng = np.random.default_rng(0)
        geoms = export_like_pipeline(v2a, train, gt, VN, work, pipeline_argv, rng, args.lf_mm, args.hf_mm)
        k = v2a.choose_image_scale(geoms, pargs.image_scale, pargs.max_image_side)
        v2a.compute_harmonisation_gains(geoms, pargs.harmonize)
        exr = v2a.ExrWriter()
        v2a.cross_view_consensus(geoms, pargs)
        written = v2a.export_frames(geoms, k, pipe, pargs, exr)
        v2a.save_sfmdata(v2a.build_sfmdata(geoms, k, written, pargs.sfm_version, [],
                                           pargs.pixel_center == "aligned"), pipe / "sfm.sfm")
        av = v2a.AliceVision(args.av_root / "bin", verbose="warning")
        v2a.process_capture([], None, None, pipe, pargs, av)

    av_mesh = pipe / "texturedMesh" / "texturedMesh.obj"
    cams = io.load_sfm_cameras(pipe / "sfm.sfm", pipe / "undistorted")

    variants = {
        "pyramid_full": tp.Options(texture_side=args.texture_side),
        "pyramid_no_align": tp.Options(texture_side=args.texture_side, align=False),
        "pyramid_mid": tp.Options(texture_side=args.texture_side, band_sharpness=(16.0, 10.0, 4.0, 2.0, 1.0, 1.0)),
        "pyramid_sharp": tp.Options(texture_side=args.texture_side, band_sharpness=(32.0, 16.0, 4.0, 2.0, 1.0, 1.0)),
        "pyramid_sharp_no_align": tp.Options(texture_side=args.texture_side, align=False,
                                             band_sharpness=(32.0, 16.0, 4.0, 2.0, 1.0, 1.0)),
        "pyramid_no_gain_spec": tp.Options(texture_side=args.texture_side, gain=False, specular=False),
        "best_view_only": tp.Options(texture_side=args.texture_side, band_sharpness=(40.0,), align=False,
                                     gain=False, specular=False),
        "plain_average": tp.Options(texture_side=args.texture_side, band_sharpness=(1.0,), align=False,
                                    gain=False, specular=False),
    }
    chosen = list(variants) if args.variants == "all" else args.variants.split(",")
    results_path = work / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    meshes = {"aliceVision_v3": av_mesh}
    for name in chosen:
        out = work / f"tex_{name}"
        if not (args.reuse and (out / av_mesh.name).exists()):
            tp.retexture(av_mesh, cams, out, variants[name])
        meshes[name] = out / av_mesh.name

    # eye centres: tracked-mesh vertices with the darkest texture around the eyes are
    # hard to pick generically; use the two mesh points that project on the iris in the
    # frontal training view instead (found once by hand for this subject).
    eye_pts = np.array(json.loads((REPO / "experiments" / "multiface_eyes.json").read_text())["eye_centres_mm"])
    from PIL import Image

    for name, path in meshes.items():
        per_view = []
        for cam in held:
            photo = np.asarray(Image.open(cam.image_path).convert("RGB"))
            render, fg = render_textured(path, cam)
            import cv2

            mask = cv2.erode(fg.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            m_all = compare(render, photo, mask)
            em = eye_mask(cam, gt, mask, eye_pts)
            m_eye = compare(render, photo, em) if em.sum() > 500 else {}
            per_view.append({"view": cam.view_id, "all": m_all, "eyes": m_eye})
            if cam.view_id in (400030, 400060, 400004):
                crop = render.astype(np.uint8)
                Image.fromarray(crop).save(work / f"render_{name}_{cam.view_id}.png")
        agg = {}
        for region in ("all", "eyes"):
            keys = per_view[0][region].keys() if per_view[0][region] else []
            agg[region] = {k: float(np.mean([p[region][k] for p in per_view if p[region]])) for k in keys}
        results[name] = {"mean": agg, "per_view": per_view}
        log(f"{name:>22}: " + "  ".join(f"{k}={v:.3f}" for k, v in agg["all"].items())
            + " | eyes " + "  ".join(f"{k}={v:.3f}" for k, v in agg["eyes"].items()))
    results_path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
