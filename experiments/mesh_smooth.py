#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bounded, non-shrinking mesh smoothing -- and the roughness metrics used by the ablations.

TESTED, NOT ADOPTED: on the Multiface ablation (geometry_ablation.py) 10-40 iterations on
top of aliceVision_meshDenoising changed the normal error by +0.01..+0.02 deg (i.e. not at
all) and raised the p90 dihedral angle.  The bumps left after denoising are mid-frequency
(several mm), caused by the views' depth maps disagreeing; they have to be fixed before
meshing (see --pixel-center / --consensus-* in vggt_omega_to_alicevision.py), not by
smoothing the mesh.  Kept because the experiments use taubin_smooth() to build their
reference surface and roughness_report() to score meshes.

aliceVision_meshFiltering smooths with plain Laplacian iterations: every iteration pulls
each vertex towards its neighbours' centroid, which removes bumps but also SHRINKS the
surface (convex regions such as the nose tip, lips and chin move inwards) and slides
vertices tangentially.  For pre-operative simulation that is the wrong trade-off: we want
less high-frequency relief without moving the anatomy.

This implements Taubin's lambda|mu smoothing (Taubin 1995), a two-step low-pass filter
with no shrinkage, with two extra safety rails:

*   updates are projected on the vertex normal, so vertices only move along the surface
    normal (no tangential sliding, triangle quality is preserved);
*   the total displacement of every vertex from the input is clamped to
    --max-displacement (in units of the median edge length, or in scene units with
    --max-displacement-abs), so the smoothing can never move the surface further than a
    known, reported bound.

Boundary vertices (open mesh borders, e.g. the neck cut) are kept fixed.

    python mesh_smooth.py mesh/filteredMesh.obj mesh/smoothedMesh.obj --iterations 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from av_mesh_io import load_obj, save_obj, vertex_normals  # noqa: E402


def umbrella_operator(F: np.ndarray, n: int):
    """Row-normalised adjacency (uniform weights) as a scipy CSR matrix, and boundary mask."""
    import scipy.sparse as sp

    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(e, axis=1)
    key = und[:, 0] * n + und[:, 1]
    uniq, counts = np.unique(key, return_counts=True)
    boundary_edges = uniq[counts == 1]
    boundary = np.zeros(n, bool)
    boundary[boundary_edges // n] = True
    boundary[boundary_edges % n] = True
    i = uniq // n
    j = uniq % n
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(deg, 1))
    return (Dinv @ A).tocsr(), boundary, deg


def median_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    e = np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2)
    return float(np.median(e))


def taubin_smooth(
    V: np.ndarray,
    F: np.ndarray,
    iterations: int = 20,
    lam: float = 0.5,
    mu: float = -0.53,
    normal_only: bool = True,
    max_displacement: float | None = None,
) -> np.ndarray:
    """Taubin lambda|mu smoothing, optionally normal-projected and displacement-clamped."""
    W, boundary, deg = umbrella_operator(F, len(V))
    free = (~boundary) & (deg > 0)
    V0 = V.astype(np.float64)
    X = V0.copy()
    for _ in range(iterations):
        for step in (lam, mu):
            delta = W @ X - X
            if normal_only:
                N = vertex_normals(X, F)
                delta = np.einsum("ij,ij->i", delta, N)[:, None] * N
            X[free] += step * delta[free]
            if max_displacement is not None:
                d = X - V0
                norm = np.linalg.norm(d, axis=1)
                over = norm > max_displacement
                X[over] = V0[over] + d[over] * (max_displacement / norm[over])[:, None]
    return X


def roughness_report(V: np.ndarray, F: np.ndarray) -> dict:
    """Dihedral-angle statistics between adjacent faces (degrees) -- the quantity that
    shows up as 'bumpy' shading."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    und = np.sort(e, axis=1)
    key = und[:, 0] * len(V) + und[:, 1]
    order = np.argsort(key, kind="stable")
    k = key[order]
    same = np.nonzero(k[1:] == k[:-1])[0]
    a, b = fid[order][same], fid[order][same + 1]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fn[a], fn[b]), -1, 1)))
    return {
        "dihedral_median_deg": float(np.median(ang)),
        "dihedral_p90_deg": float(np.percentile(ang, 90)),
        "frac_over_10deg": float(np.mean(ang > 10)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=-0.53)
    parser.add_argument("--max-displacement", type=float, default=1.0,
                        help="clamp in units of the median edge length (0 = no clamp)")
    parser.add_argument("--max-displacement-abs", type=float, default=0.0,
                        help="clamp in scene units; overrides --max-displacement when > 0")
    parser.add_argument("--allow-tangential", action="store_true",
                        help="do not project updates on the normal (classic Taubin)")
    args = parser.parse_args(argv)

    mesh = load_obj(args.input)
    edge = median_edge_length(mesh.V, mesh.F)
    cap = args.max_displacement_abs if args.max_displacement_abs > 0 else (
        args.max_displacement * edge if args.max_displacement > 0 else None)
    before = roughness_report(mesh.V, mesh.F)
    V = taubin_smooth(mesh.V, mesh.F, args.iterations, args.lam, args.mu,
                      normal_only=not args.allow_tangential, max_displacement=cap)
    after = roughness_report(V, mesh.F)
    disp = np.linalg.norm(V - mesh.V, axis=1)
    mesh.V = V
    save_obj(args.output, mesh)
    report = {"median_edge": edge, "cap": cap, "before": before, "after": after,
              "displacement_median": float(np.median(disp)), "displacement_max": float(disp.max())}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
