#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local side-by-side mesh comparison viewer.

    python viewer/serve.py --root C:/atop/data
    python viewer/serve.py --root /mnt/c/atop/data --root /tmp/mf_eval --port 8800

Scans every --root for .obj files, serves viewer/index.html, and converts a mesh to a
compact indexed binary the first time it is opened (numpy-parsed, smooth normals computed
on the original vertices so UV seams do not show up as shading seams).  Textures are
served straight from disk.  Only listens on 127.0.0.1 unless --host says otherwise.

Needs numpy (already in the pipeline's environment).  No internet access is needed:
three.js is vendored in viewer/vendor.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from av_mesh_io import load_obj, parse_mtl, vertex_normals  # noqa: E402

SKIP_DIRS = {".git", "node_modules", "__pycache__", "depthMaps", "depthMapsRaw", "depthMapsFiltered", "undistorted"}


class Library:
    def __init__(self, roots: list[Path], cache_size: int = 6) -> None:
        self.roots = [r.resolve() for r in roots]
        self.models: list[dict] = []
        self.cache: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()
        self.scan()

    def scan(self) -> list[dict]:
        found = []
        for ri, root in enumerate(self.roots):
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
                for name in sorted(filenames):
                    if not name.lower().endswith(".obj"):
                        continue
                    path = Path(dirpath) / name
                    try:
                        st = path.stat()
                        with open(path, "rb") as fh:
                            head = fh.read(1 << 16)
                    except OSError:
                        continue
                    rel = path.relative_to(root).as_posix()
                    found.append({
                        "id": f"{ri}:{rel}",
                        "root": str(root),
                        "rel": rel,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                        "textured": b"mtllib" in head and b"\nvt " in head,
                    })
        self.models = found
        return found

    def resolve(self, model_id: str) -> Path:
        ri, rel = model_id.split(":", 1)
        root = self.roots[int(ri)]
        path = (root / rel).resolve()
        if root not in path.parents:
            raise PermissionError(model_id)
        return path

    def file_url(self, path: Path) -> str | None:
        for ri, root in enumerate(self.roots):
            if root in path.parents and path.is_file():
                return f"/files/{ri}/" + urllib.parse.quote(path.relative_to(root).as_posix())
        return None

    def mesh_bytes(self, model_id: str) -> bytes:
        path = self.resolve(model_id)
        mtime = path.stat().st_mtime
        with self.lock:
            hit = self.cache.get(model_id)
            if hit and hit[0] == mtime:
                self.cache.move_to_end(model_id)
                return hit[1]
        data = self.convert(path)
        with self.lock:
            self.cache[model_id] = (mtime, data)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return data

    def convert(self, path: Path) -> bytes:
        t0 = time.time()
        mesh = load_obj(path)
        V, F = mesh.V, mesh.F
        normals = vertex_normals(V, F)  # on the original vertices: no seams at UV cuts
        has_uv = mesh.VT is not None and mesh.FT is not None and len(mesh.FT) == len(F)

        # order faces by material so each material is one draw group
        mat = mesh.face_material if mesh.face_material is not None else np.zeros(len(F), np.int32)
        order = np.argsort(mat, kind="stable")
        F = F[order]
        mat = mat[order]
        if has_uv:
            FT = mesh.FT[order]
            key = F.reshape(-1).astype(np.int64) * (len(mesh.VT) + 1) + FT.reshape(-1)
            uniq, inv = np.unique(key, return_inverse=True)
            vi = uniq // (len(mesh.VT) + 1)
            ti = uniq % (len(mesh.VT) + 1)
            pos, nrm, uv = V[vi], normals[vi], mesh.VT[ti]
            idx = inv.reshape(-1, 3)
        else:
            pos, nrm, uv, idx = V, normals, None, F

        textures: dict[str, str] = {}
        if mesh.mtllib and (path.parent / mesh.mtllib).is_file():
            textures = parse_mtl(path.parent / mesh.mtllib)
        groups = []
        for m in np.unique(mat):
            sel = np.nonzero(mat == m)[0]
            name = mesh.materials[m] if mesh.materials else ""
            tex = textures.get(name)
            url = self.file_url((path.parent / tex).resolve()) if (tex and has_uv) else None
            groups.append({"start": int(sel[0]) * 3, "count": int(len(sel)) * 3, "material": name, "texture": url})

        header = {
            "nv": int(len(pos)), "nf": int(len(idx)), "hasUV": bool(has_uv),
            "groups": groups,
            "bboxMin": V.min(0).tolist(), "bboxMax": V.max(0).tolist(),
            "vertices": int(len(V)), "path": str(path),
        }
        hj = json.dumps(header).encode("utf-8")
        hj += b" " * (-len(hj) % 4)
        parts = [b"FMV1", struct.pack("<I", len(hj)), hj,
                 pos.astype(np.float32).tobytes(), nrm.astype(np.float32).tobytes()]
        if uv is not None:
            parts.append(uv.astype(np.float32).tobytes())
        parts.append(idx.astype(np.uint32).tobytes())
        data = b"".join(parts)
        print(f"[viewer] converted {path} ({len(F):,} faces) in {time.time() - t0:.1f}s", flush=True)
        return data


def make_handler(lib: Library):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE), **kw)

        def log_message(self, fmt, *args):  # quiet: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def send_bytes(self, data: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            try:
                if url.path == "/api/models":
                    models = lib.scan() if q.get("refresh") else lib.models
                    body = json.dumps({"roots": [str(r) for r in lib.roots], "models": models})
                    return self.send_bytes(body.encode("utf-8"), "application/json; charset=utf-8")
                if url.path == "/api/mesh":
                    return self.send_bytes(lib.mesh_bytes(q["id"][0]), "application/octet-stream")
                if url.path.startswith("/files/"):
                    _, _, ri, rel = url.path.split("/", 3)
                    root = lib.roots[int(ri)]
                    path = (root / urllib.parse.unquote(rel)).resolve()
                    if root not in path.parents or not path.is_file():
                        return self.send_bytes(b"not found", "text/plain", 404)
                    ctype = self.guess_type(str(path))
                    return self.send_bytes(path.read_bytes(), ctype)
            except PermissionError:
                return self.send_bytes(b"forbidden", "text/plain", 403)
            except Exception as exc:  # report conversion errors to the page
                return self.send_bytes(f"{type(exc).__name__}: {exc}".encode("utf-8"), "text/plain", 500)
            return super().do_GET()

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, action="append", default=[],
                    help="folder to scan for .obj files (repeatable; default: current folder)")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    roots = args.root or [Path.cwd()]
    missing = [str(r) for r in roots if not r.is_dir()]
    if missing:
        raise SystemExit(f"not a folder: {', '.join(missing)}")
    lib = Library(roots)
    print(f"[viewer] {len(lib.models)} .obj files under {', '.join(str(r) for r in lib.roots)}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(lib))
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    print(f"[viewer] open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
