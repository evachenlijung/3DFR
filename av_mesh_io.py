#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small, dependency-light helpers shared by texture_pyramid.py and mesh_smooth.py.

    * OBJ / MTL reading and writing (keeps AliceVision's UDIM materials intact)
    * AliceVision SfMData camera loading (the exact inverse of build_sfmdata() in
      vggt_omega_to_alicevision.py, so it reads back what that script wrote)
    * a numpy triangle rasteriser (z-buffer + triangle id + barycentrics), used both
      to render a mesh into a camera and to rasterise a UV atlas
    * push-pull hole filling (Gaussian-pyramid normalised convolution)

Only numpy, opencv-python and pillow are required -- no OpenGL/EGL, so it runs the same
under WSL, a headless server or Windows.

Pixel convention: pixel (x, y) has its centre at integer coordinates, the same
"integer index, no +0.5" convention the AliceVision exporter uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# OBJ / MTL
# --------------------------------------------------------------------------------------


@dataclass
class ObjMesh:
    V: np.ndarray  # (n, 3) float64
    F: np.ndarray  # (f, 3) int64, 0-based
    VT: np.ndarray | None = None  # (m, 2) float64
    FT: np.ndarray | None = None  # (f, 3) int64
    face_material: np.ndarray | None = None  # (f,) int32 index into materials
    materials: list[str] = field(default_factory=list)
    mtllib: str | None = None


def _parse_face_block(lines: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Fast path for uniform triangle 'f' lines; falls back to a fan-triangulating loop."""
    if not lines:
        return np.zeros((0, 3), np.int64), None
    first = lines[0].split()[1:]
    per_vertex = first[0].count("/") + 1
    uniform = len(first) == 3 and "//" not in lines[0]
    if uniform:
        try:
            flat = " ".join(line[2:] for line in lines).replace("/", " ").split()
            data = np.asarray(flat, dtype=np.int64).reshape(len(lines), 3, per_vertex)
            F = data[:, :, 0] - 1
            FT = data[:, :, 1] - 1 if per_vertex >= 2 else None
            return F, FT
        except ValueError:
            pass
    faces, tex = [], []
    for line in lines:
        corners = line.split()[1:]
        vi = [int(c.split("/")[0]) - 1 for c in corners]
        ti = [int(c.split("/")[1]) - 1 if c.count("/") >= 1 and c.split("/")[1] else -1 for c in corners]
        for j in range(1, len(corners) - 1):
            faces.append([vi[0], vi[j], vi[j + 1]])
            tex.append([ti[0], ti[j], ti[j + 1]])
    F = np.asarray(faces, np.int64)
    FT = np.asarray(tex, np.int64)
    return F, (FT if (FT >= 0).all() else None)


def load_obj(path: Path) -> ObjMesh:
    V, VT = [], []
    face_lines: list[str] = []
    face_mat: list[int] = []
    materials: list[str] = []
    current = -1
    mtllib = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                V.append(line)
            elif line.startswith("vt "):
                VT.append(line)
            elif line.startswith("f "):
                face_lines.append(line.strip())
                face_mat.append(current)
            elif line.startswith("usemtl"):
                name = line.split(maxsplit=1)[1].strip()
                if name not in materials:
                    materials.append(name)
                current = materials.index(name)
            elif line.startswith("mtllib"):
                mtllib = line.split(maxsplit=1)[1].strip()

    def to_array(rows: list[str], cols: int) -> np.ndarray:
        if not rows:
            return np.zeros((0, cols))
        flat = " ".join(r[2:] if r[1] == " " else r[3:] for r in rows).split()
        arr = np.asarray(flat, dtype=np.float64)
        width = arr.size // len(rows)
        return arr.reshape(len(rows), width)[:, :cols]

    Vn = to_array(V, 3)
    VTn = to_array(VT, 2) if VT else None
    F, FT = _parse_face_block(face_lines)
    fm = np.asarray(face_mat, np.int32)
    if len(fm) != len(F):  # polygons were fan-triangulated; material per triangle is lost
        fm = np.zeros(len(F), np.int32)
    if (fm < 0).any():
        if not materials:
            materials.append("default")
        fm[fm < 0] = 0
    return ObjMesh(Vn, F, VTn, FT, fm, materials, mtllib)


def save_obj(path: Path, mesh: ObjMesh) -> None:
    """Writes v / vt / f (+ usemtl groups).  Enough for AliceVision and every DCC tool."""
    path = Path(path)
    lines: list[str] = []
    if mesh.mtllib:
        lines.append(f"mtllib {mesh.mtllib}")
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in mesh.V)
    has_uv = mesh.VT is not None and mesh.FT is not None
    if has_uv:
        lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in mesh.VT)
    order = np.argsort(mesh.face_material, kind="stable") if mesh.face_material is not None else np.arange(len(mesh.F))
    current = None
    for fi in order:
        if mesh.face_material is not None and mesh.materials:
            m = int(mesh.face_material[fi])
            if m != current:
                lines.append(f"usemtl {mesh.materials[m]}")
                current = m
        a, b, c = mesh.F[fi] + 1
        if has_uv:
            ta, tb, tc = mesh.FT[fi] + 1
            lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
        else:
            lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_mtl(path: Path) -> dict[str, str]:
    """material name -> map_Kd file name (relative to the .mtl)."""
    out: dict[str, str] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("newmtl"):
            current = line.split(maxsplit=1)[1].strip()
        elif line.startswith("map_Kd") and current is not None:
            out[current] = line.split()[-1]
    return out


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for i in range(3):
        np.add.at(N, F[:, i], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)


# --------------------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------------------


@dataclass
class Camera:
    view_id: int
    K: np.ndarray  # 3x3, pixel-centre-at-integer convention
    R: np.ndarray  # camera-from-world rotation (OpenCV: x right, y down, z forward)
    t: np.ndarray  # camera-from-world translation
    width: int
    height: int
    image_path: Path | None = None

    @property
    def C(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * Xc[:, 0] / z + self.K[0, 1] * Xc[:, 1] / z + self.K[0, 2]
            v = self.K[1, 1] * Xc[:, 1] / z + self.K[1, 2]
        return u, v, z

    def scaled(self, s: float) -> "Camera":
        """Same camera for an image resized by s (centre-aligned resize, as cv2/PIL do)."""
        S = np.array([[s, 0, 0.5 * s - 0.5], [0, s, 0.5 * s - 0.5], [0, 0, 1.0]])
        return Camera(self.view_id, S @ self.K, self.R, self.t,
                      int(round(self.width * s)), int(round(self.height * s)), self.image_path)


def load_sfm_cameras(sfm_path: Path, images_dir: Path | None = None) -> list[Camera]:
    """Read an SfMData JSON as written by vggt_omega_to_alicevision.build_sfmdata().

    Inverse of that writer:
        rotation   column-major R (camera-from-world), center = -R^T t
        focal      legacy (< 1.2.11): fx = f_mm * W / sensorWidth, fy = fx * pixelRatio
                   newer            : fy = f_mm * W / sensorWidth, fx = fy / pixelRatio
        principal  stored as an offset from the image centre (W/2, H/2)
    """
    sfm = json.loads(Path(sfm_path).read_text(encoding="utf-8"))
    version = tuple(int(v) for v in sfm.get("version", ["1", "2", "6"]))
    legacy = version < (1, 2, 11)
    intrinsics = {i["intrinsicId"]: i for i in sfm["intrinsics"]}
    poses = {p["poseId"]: p for p in sfm["poses"]}
    cams: list[Camera] = []
    for view in sfm["views"]:
        intr = intrinsics[view["intrinsicId"]]
        pose = poses[view["poseId"]]["pose"]["transform"]
        W, H = int(view["width"]), int(view["height"])
        sensor = float(intr["sensorWidth"])
        f = float(intr["focalLength"]) * W / sensor
        ratio = float(intr.get("pixelRatio", 1.0))
        fx, fy = (f, f * ratio) if legacy else (f / ratio, f)
        pp = [float(v) for v in intr.get("principalPoint", ["0", "0"])]
        K = np.array([[fx, 0, W / 2.0 + pp[0]], [0, fy, H / 2.0 + pp[1]], [0, 0, 1.0]])
        R = np.asarray([float(v) for v in pose["rotation"]]).reshape(3, 3, order="F")
        C = np.asarray([float(v) for v in pose["center"]])
        path = Path(view["path"])
        if images_dir is not None:
            for candidate in (images_dir / path.name, images_dir / f"{view['viewId']}{path.suffix}"):
                if candidate.is_file():
                    path = candidate
                    break
        cams.append(Camera(int(view["viewId"]), K, R, -R @ C, W, H, path))
    cams.sort(key=lambda c: c.view_id)
    return cams


# --------------------------------------------------------------------------------------
# rasteriser
# --------------------------------------------------------------------------------------


def rasterize(
    xy: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    depth: np.ndarray | None = None,
    chunk: int = 6_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise triangles whose vertices sit at pixel coordinates `xy` (n, 2).

    With `depth` (n,) a z-buffer keeps the nearest surface; without it (UV atlases, which
    do not overlap) any covering triangle wins.

    Returns
        zbuf  (h, w) float32, +inf where empty   (only meaningful with depth)
        tri   (h, w) int32, -1 where empty
        bary  (h, w, 2) float32 -- weights of the triangle's 2nd and 3rd vertex
    Depth and barycentrics are interpolated affinely in screen space, which for the
    sub-millimetre triangles of a face scan is indistinguishable from perspective-correct.
    """
    tri_xy = xy[faces]  # (f, 3, 2)
    x0, y0 = tri_xy[:, 0, 0], tri_xy[:, 0, 1]
    x1, y1 = tri_xy[:, 1, 0], tri_xy[:, 1, 1]
    x2, y2 = tri_xy[:, 2, 0], tri_xy[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    xmin = np.maximum(np.ceil(np.min(tri_xy[:, :, 0], 1) - 1e-6), 0).astype(np.int64)
    xmax = np.minimum(np.floor(np.max(tri_xy[:, :, 0], 1) + 1e-6), width - 1).astype(np.int64)
    ymin = np.maximum(np.ceil(np.min(tri_xy[:, :, 1], 1) - 1e-6), 0).astype(np.int64)
    ymax = np.minimum(np.floor(np.max(tri_xy[:, :, 1], 1) + 1e-6), height - 1).astype(np.int64)
    ok = (xmax >= xmin) & (ymax >= ymin) & (np.abs(area) > 1e-12) & np.isfinite(area)
    if depth is not None:
        zf = depth[faces]
        ok &= (zf > 0).all(1)

    zbuf = np.full(height * width, np.inf, np.float32)
    tri = np.full(height * width, -1, np.int32)
    bary = np.zeros((height * width, 2), np.float32)

    ids = np.nonzero(ok)[0]
    extent = np.maximum(xmax - xmin + 1, ymax - ymin + 1)[ids]
    order = np.argsort(extent, kind="stable")
    ids, extent = ids[order], extent[order]

    def fragments(sel: np.ndarray, size: int):
        g = np.arange(size)
        gx = np.tile(g, size)
        gy = np.repeat(g, size)
        px = xmin[sel, None] + gx[None, :]
        py = ymin[sel, None] + gy[None, :]
        inside = (px <= xmax[sel, None]) & (py <= ymax[sel, None])
        a = area[sel, None]
        dx, dy = px - x0[sel, None], py - y0[sel, None]
        w1 = (dx * (y2[sel, None] - y0[sel, None]) - (x2[sel, None] - x0[sel, None]) * dy) / a
        w2 = ((x1[sel, None] - x0[sel, None]) * dy - dx * (y1[sel, None] - y0[sel, None])) / a
        w0 = 1.0 - w1 - w2
        eps = -1e-7
        inside &= (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        r, c = np.nonzero(inside)
        pix = py[r, c] * width + px[r, c]
        face = sel[r]
        b1, b2 = w1[r, c], w2[r, c]
        if depth is not None:
            zf_ = depth[faces[face]]
            z = (1.0 - b1 - b2) * zf_[:, 0] + b1 * zf_[:, 1] + b2 * zf_[:, 2]
        else:
            z = None
        return pix, face, b1, b2, z

    def batches():
        start = 0
        while start < len(ids):
            size = int(extent[start])
            size = 1 << int(np.ceil(np.log2(max(size, 1))))  # bucket by power of two
            stop = int(np.searchsorted(extent, size, side="right"))
            per = max(1, chunk // (size * size))
            for s in range(start, stop, per):
                yield ids[s : min(stop, s + per)], size
            start = stop

    if depth is None:
        for sel, size in batches():
            pix, face, b1, b2, _ = fragments(sel, size)
            tri[pix] = face
            bary[pix, 0] = b1
            bary[pix, 1] = b2
    else:
        # pass 1: nearest depth per pixel; pass 2: the fragment that produced it
        for sel, size in batches():
            pix, _, _, _, z = fragments(sel, size)
            np.minimum.at(zbuf, pix, z.astype(np.float32))
        for sel, size in batches():
            pix, face, b1, b2, z = fragments(sel, size)
            win = z.astype(np.float32) <= zbuf[pix]
            tri[pix[win]] = face[win]
            bary[pix[win], 0] = b1[win]
            bary[pix[win], 1] = b2[win]
    return zbuf.reshape(height, width), tri.reshape(height, width), bary.reshape(height, width, 2)


def interpolate(attr: np.ndarray, faces: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Barycentric interpolation of a per-vertex attribute at rasterised samples."""
    f = faces[tri]
    b1, b2 = bary[..., 0:1].astype(np.float64), bary[..., 1:2].astype(np.float64)
    return (1.0 - b1 - b2) * attr[f[..., 0]] + b1 * attr[f[..., 1]] + b2 * attr[f[..., 2]]


# --------------------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------------------


def push_pull_fill(image: np.ndarray, mask: np.ndarray, levels: int = 12) -> np.ndarray:
    """Fill ~mask pixels with a smooth continuation of the masked ones (normalised
    convolution on a Gaussian pyramid).  Keeps masked pixels exactly."""
    import cv2

    img = image.astype(np.float32)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    m = mask.astype(np.float32)
    pyr_v, pyr_m = [img * m[..., None]], [m]
    for _ in range(levels):
        h, w = pyr_m[-1].shape
        if min(h, w) < 2:
            break
        v = cv2.pyrDown(pyr_v[-1])
        mm = cv2.pyrDown(pyr_m[-1])
        if v.ndim == 2:
            v = v[..., None]
        pyr_v.append(v)
        pyr_m.append(mm)
    # coarsest level: normalise
    cur = pyr_v[-1] / np.maximum(pyr_m[-1][..., None], 1e-8)
    for lvl in range(len(pyr_v) - 2, -1, -1):
        h, w = pyr_m[lvl].shape
        up = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[..., None]
        norm = pyr_v[lvl] / np.maximum(pyr_m[lvl][..., None], 1e-8)
        alpha = np.clip(pyr_m[lvl], 0.0, 1.0)[..., None]
        cur = alpha * norm + (1.0 - alpha) * up
    out = np.where(mask[..., None] > 0, img, cur)
    return out[..., 0] if squeeze else out


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb8(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    s = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return np.clip(s * 255.0 + 0.5, 0, 255).astype(np.uint8)


def remap_points(image: np.ndarray, x: np.ndarray, y: np.ndarray, interpolation: int | None = None) -> np.ndarray:
    """Bilinear sample image at arbitrary (x, y) arrays of any length (cv2.remap wants
    2-D maps no wider than 32767, so the points are folded into rows)."""
    import cv2

    interpolation = cv2.INTER_LINEAR if interpolation is None else interpolation
    n = x.size
    if n == 0:
        shape = (0,) + image.shape[2:]
        return np.zeros(shape, np.float32)
    cols = 4096
    rows = (n + cols - 1) // cols
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n] = x
    my[:n] = y
    out = cv2.remap(image, mx.reshape(rows, cols), my.reshape(rows, cols), interpolation,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.reshape((rows * cols,) + out.shape[2:])[:n]


AV_FLIP = np.diag([1.0, -1.0, -1.0])


def align_mesh_to_cameras(V: np.ndarray, cams: Sequence[Camera]) -> tuple[np.ndarray, bool]:
    """AliceVision writes its meshes in a y/z-flipped frame (x, -y, -z) relative to the
    SfMData cameras (OpenCV convention).  Return the vertices in the camera frame and
    whether a flip was applied, deciding by which hypothesis puts more vertices in front
    of the cameras and inside their images."""

    def score(X: np.ndarray) -> float:
        sample = X[:: max(1, len(X) // 20000)]
        total = 0.0
        for cam in cams:
            u, v, z = cam.project(sample)
            total += np.mean((z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height))
        return total

    flipped = V @ AV_FLIP
    if score(flipped) > score(V):
        return flipped, True
    return V, False
