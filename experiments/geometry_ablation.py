#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bumpiness ablation on Multiface (same 7 eye-level + 2 under-chin layout as the rig).

Every configuration re-runs this repository's export code (consensus -> upsampling ->
EXR/SfM) on the SAME simulated VGGT depth maps (fixed seed), then the real
aliceVision_meshing / meshFiltering / meshDenoising, then optionally mesh_smooth.py, and
compares the mesh with the reference surface the depth maps were rendered from:

    dist_*      |signed distance| to the reference surface, mm        (accuracy)
    nerr_*      angle between mesh normal and reference normal, deg  (what shading shows
                as bumps; the most direct "bumpiness" number)
    dihedral_*  angle between adjacent faces, deg                    (intrinsic roughness)

Only the face region is scored (reference vertices within 85 mm of the nose tip that at
least two training views see), so the open neck / scalp borders do not dominate.
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
sys.path.insert(0, str(REPO / "experiments"))

import av_mesh_io as io  # noqa: E402
import mesh_smooth  # noqa: E402
import multiface_eval as me  # noqa: E402

V4 = ["--depth-upsample", "cubic", "--pixel-center", "aligned",
      "--consensus-sampling", "bilinear", "--consensus-fuse", "inlier-mean"]
CONFIGS = {
    "v3": [],
    "v3+cubic": ["--depth-upsample", "cubic"],
    "v3+aligned": ["--pixel-center", "aligned"],
    "v3+bilinear": ["--consensus-sampling", "bilinear"],
    "v3+inlier-mean": ["--consensus-fuse", "inlier-mean"],
    "v4-depth": V4,
    # second round: v4-depth plus stronger consensus smoothing / mesh denoising
    "v4+smooth4": V4 + ["--consensus-smooth-radius", "4"],
    "v4+denoise5": V4 + ["--denoise-iterations", "5"],
    "v4+mrdenoise5": V4 + ["--denoise-iterations", "5", "--denoise-lambda", "2.0", "--denoise-eta", "1.8",
                           "--denoise-nu", "0.3"],
}


def face_region(gt: io.ObjMesh, VN, cams) -> np.ndarray:
    seen = np.zeros(len(gt.V), int)
    for cam in cams:
        u, v, z = cam.project(gt.V)
        d = cam.C[None] - gt.V
        facing = np.einsum("ij,ij->i", VN, d) > 0.2 * np.linalg.norm(d, axis=1)
        inside = (u > 0) & (u < cam.width - 1) & (v > 0) & (v < cam.height - 1)
        seen += facing & inside
    # nose tip: the reference vertex closest to the frontal-most training camera
    frontal = cams[3]
    nose = gt.V[np.argmin(np.linalg.norm(gt.V - frontal.C[None], axis=1))]
    return (seen >= 2) & (np.linalg.norm(gt.V - nose[None], axis=1) < 85.0)


def score(mesh_path: Path, gt: io.ObjMesh, VN, region: np.ndarray, cams) -> dict:
    from scipy.spatial import cKDTree

    m = io.load_obj(mesh_path)
    m.V, _ = io.align_mesh_to_cameras(m.V, cams)
    tree_gt = cKDTree(gt.V[region])
    d, i = tree_gt.query(m.V, distance_upper_bound=4.0)
    ok = np.isfinite(d)
    gi = np.nonzero(region)[0][i[ok]]
    mn = io.vertex_normals(m.V, m.F)
    signed = np.einsum("ij,ij->i", m.V[ok] - gt.V[gi], VN[gi])
    nerr = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", mn[ok], VN[gi])), -1, 1)))
    # faces fully inside the scored region
    keep = np.zeros(len(m.V), bool)
    keep[np.nonzero(ok)[0]] = True
    F = m.F[keep[m.F].all(1)]
    rough = mesh_smooth.roughness_report(m.V, F) if len(F) else {}
    return {
        "vertices_in_face": int(ok.sum()),
        "dist_median_mm": float(np.median(np.abs(signed))),
        "dist_p90_mm": float(np.percentile(np.abs(signed), 90)),
        "nerr_median_deg": float(np.median(nerr)),
        "nerr_p90_deg": float(np.percentile(nerr, 90)),
        **{k.replace("dihedral_", "dihedral_"): v for k, v in rough.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--multiface", type=Path, required=True)
    ap.add_argument("--expression", default="E001_Neutral_Eyes_Open")
    ap.add_argument("--frame", default="000102")
    ap.add_argument("--av-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--lf-mm", type=float, default=0.6)
    ap.add_argument("--hf-mm", type=float, default=0.15)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--taubin", type=int, nargs="*", default=[10, 20, 40],
                    help="Taubin iteration counts applied on top of each config's denoised mesh")
    args = ap.parse_args()

    os.environ["ALICEVISION_ROOT"] = str(args.av_root)
    os.environ["LD_LIBRARY_PATH"] = f"{args.av_root}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    import vggt_omega_to_alicevision as v2a

    root = args.multiface
    krt = me.load_krt(root / "KRT")
    img = lambda n: root / "images" / args.expression / n / f"{args.frame}.png"
    train = [me.multiface_camera(n, krt, img(n)) for n in me.EYE_LEVEL + me.UNDER_CHIN]
    gt = io.load_obj(root / "tracked_mesh" / args.expression / f"{args.frame}.obj")
    gt = me.subdivide(me.subdivide(gt))
    gt.V = mesh_smooth.taubin_smooth(gt.V, gt.F, iterations=10, normal_only=False)
    VN = io.vertex_normals(gt.V, gt.F)
    region = face_region(gt, VN, train)

    results_path = args.work / "geometry_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in args.configs.split(","):
        extra = CONFIGS[name]
        work = args.work / name.replace("+", "_")
        pipe = work / "pipeline"
        argv = ["--images", str(work / "photos"), "--checkpoint", "unused.pt", "--av-bin", str(args.av_root / "bin"),
                "--output", str(pipe), "--image-resolution", "512", "--skip-inference", "--export-only"] + extra
        pargs = v2a.parse_args(argv)
        rng = np.random.default_rng(0)  # identical simulated VGGT output for every config
        geoms = me.export_like_pipeline(v2a, train, gt, VN, work, argv, rng, args.lf_mm, args.hf_mm)
        k = v2a.choose_image_scale(geoms, 0, 8192)
        exr = v2a.ExrWriter()
        v2a.cross_view_consensus(geoms, pargs)
        written = v2a.export_frames(geoms, k, pipe, pargs, exr)
        v2a.save_sfmdata(v2a.build_sfmdata(geoms, k, written, pargs.sfm_version, [], pargs.pixel_center == "aligned"),
                         pipe / "sfm.sfm")
        av = v2a.AliceVision(args.av_root / "bin", verbose="error")
        mesh_dir = pipe / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        dense = mesh_dir / "densePointCloud.abc"
        av.run("aliceVision_meshing", input=pipe / "sfm.sfm", depthMapsFolder=pipe / "depthMaps", output=dense,
               outputMesh=mesh_dir / "rawMesh.obj", maxInputPoints=pargs.max_input_points, maxPoints=pargs.max_points,
               maxPointsPerVoxel=pargs.max_points_per_voxel, minAngleThreshold=pargs.min_angle_threshold,
               minStep=pargs.min_step, minVis=pargs.min_vis, simFactor=pargs.sim_factor, angleFactor=pargs.angle_factor,
               pixSizeMarginInitCoef=pargs.pix_size_margin_init_coef,
               pixSizeMarginFinalCoef=pargs.pix_size_margin_final_coef, nPixelSizeBehind=pargs.n_pixel_size_behind,
               voteMarginFactor=pargs.vote_margin_factor, contributeMarginFactor=pargs.contribute_margin_factor,
               simGaussianSizeInit=pargs.sim_gaussian_size_init, simGaussianSize=pargs.sim_gaussian_size,
               partitioning="singleBlock", repartition="multiResolution", estimateSpaceFromSfM=False,
               addLandmarksToTheDensePointCloud=False, colorizeOutput=False)
        av.run("aliceVision_meshFiltering", inputMesh=mesh_dir / "rawMesh.obj", outputMesh=mesh_dir / "filteredMesh.obj",
               keepLargestMeshOnly=True, smoothingIterations=pargs.smoothing_iterations,
               filteringIterations=pargs.filtering_iterations)
        av.run("aliceVision_meshDenoising", input=mesh_dir / "filteredMesh.obj", output=mesh_dir / "denoisedMesh.obj",
               denoisingIterations=pargs.denoise_iterations, meshUpdateClosenessWeight=0.001,
               **{"lambda": pargs.denoise_lambda}, eta=pargs.denoise_eta, mu=pargs.denoise_mu, nu=pargs.denoise_nu,
               meshUpdateMethod=0)
        res = {"raw": score(mesh_dir / "rawMesh.obj", gt, VN, region, train),
               "denoised": score(mesh_dir / "denoisedMesh.obj", gt, VN, region, train)}
        for it in args.taubin:
            out = mesh_dir / f"taubin{it}.obj"
            mesh_smooth.main([str(mesh_dir / "denoisedMesh.obj"), str(out), "--iterations", str(it)])
            res[f"taubin{it}"] = score(out, gt, VN, region, train)
        results[name] = res
        for stage, r in res.items():
            print(f"[geom] {name:>15} {stage:>9}: " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                              for k, v in r.items()), flush=True)
        results_path.write_text(json.dumps(results, indent=1))
        shutil.rmtree(work / "photos", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
