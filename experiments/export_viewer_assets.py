#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack meshes (+ textures) into a compact binary the web viewer (viewer/index.html) loads.

Format of <name>.mesh (little endian):
    char[4]  "FMSH"
    uint32   n_vertices, n_faces, has_uv
    float32  bbox_min[3], bbox_size[3]
    uint16   positions[n_vertices * 3]   (quantised in the bbox)
    uint16   uv[n_vertices * 2]          (only if has_uv; 0..65535 -> 0..1)
    padding  to a multiple of 4 bytes
    uint32   indices[n_faces * 3]

Vertices are expressed in a display frame built from one camera (x right, y up, looking
down -z) and cropped to a sphere around the face, so the files stay well under the
artifact size limits while keeping every original triangle inside the crop.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import av_mesh_io as io  # noqa: E402
import multiface_eval as me  # noqa: E402


def display_frame(cam: io.Camera, centre: np.ndarray):
    """world -> display: camera axes, y up, origin at `centre`."""
    M = np.diag([1.0, -1.0, -1.0]) @ cam.R
    return lambda X: (X - centre) @ M.T


def pack(path: Path, V: np.ndarray, F: np.ndarray, UV: np.ndarray | None) -> int:
    lo = V.min(0)
    size = np.maximum(V.max(0) - lo, 1e-9)
    q = np.round((V - lo) / size * 65535).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(b"FMSH")
        f.write(struct.pack("<3I", len(V), len(F), int(UV is not None)))
        f.write(struct.pack("<6f", *lo, *size))
        f.write(q.tobytes())
        if UV is not None:
            f.write(np.round(np.clip(UV, 0, 1) * 65535).astype(np.uint16).tobytes())
        f.write(b"\0" * (-f.tell() % 4))  # indices start 4-byte aligned
        f.write(F.astype(np.uint32).tobytes())
    return path.stat().st_size


def crop_and_split(mesh: io.ObjMesh, to_display, radius: float):
    V = to_display(mesh.V)
    keep_v = np.linalg.norm(V - np.array([0, 0, 0]), axis=1) < radius
    fk = keep_v[mesh.F].all(1)
    F = mesh.F[fk]
    if mesh.VT is not None and mesh.FT is not None:
        FT = mesh.FT[fk]
        pairs = np.stack([F.reshape(-1), FT.reshape(-1)], 1)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        Vn = V[uniq[:, 0]]
        uv = mesh.VT[uniq[:, 1]]
        uv = uv - np.floor(uv)  # single UDIM tile expected in the viewer
        return Vn, inv.reshape(-1, 3), uv
    used, inv = np.unique(F.reshape(-1), return_inverse=True)
    return V[used], inv.reshape(-1, 3), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiface", type=Path, required=True)
    ap.add_argument("--camera", default="400016", help="camera defining the display frame")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=115.0, help="crop radius around the face centre, mm")
    ap.add_argument("--mesh", action="append", default=[], help="name=path.obj (repeatable)")
    ap.add_argument("--texture", action="append", default=[], help="name=path.png (repeatable)")
    ap.add_argument("--texture-size", type=int, default=4096)
    args = ap.parse_args()

    from PIL import Image

    krt = me.load_krt(args.multiface / "KRT")
    img = args.multiface / "images" / "E001_Neutral_Eyes_Open" / args.camera / "000102.png"
    cam = me.multiface_camera(args.camera, krt, img)
    # face centre: 330 mm... use the point the frontal camera looks at, at the eyes' depth
    import json

    eyes = np.array(json.loads((Path(__file__).parent / "multiface_eyes.json").read_text())["eye_centres_mm"])
    centre = eyes.mean(0) + cam.R.T @ np.array([0.0, 25.0, 30.0])  # a bit lower and deeper: nose/mouth
    to_display = display_frame(cam, centre)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.mesh:
        name, path = spec.split("=", 1)
        mesh = io.load_obj(Path(path))
        mesh.V, flipped = io.align_mesh_to_cameras(mesh.V, [cam])
        V, F, UV = crop_and_split(mesh, to_display, args.radius)
        n = pack(args.out / f"{name}.mesh", V, F, UV)
        print(f"{name}: {len(V):,} v {len(F):,} f {n / 1e6:.1f} MB (flip {flipped})")
    for spec in args.texture:
        name, path = spec.split("=", 1)
        im = Image.open(path).convert("RGB")
        if max(im.size) > args.texture_size:
            im = im.resize((args.texture_size, args.texture_size), Image.Resampling.LANCZOS)
        im.save(args.out / f"{name}.jpg", quality=90, subsampling=0)
        print(f"{name}.jpg {(args.out / f'{name}.jpg').stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
