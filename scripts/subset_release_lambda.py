#!/usr/bin/env python3
"""Subset the lambda axis of an assembled luhdm data release (no recompute).

Produces a new release HDF5 that is an *axis subset* of an existing one: only
the requested finite mediator ranges (plus the massless ``inf`` sentinel) are
kept on ``axes/lambda_m``, and every dataset carrying a ``lambda_m`` dimension
(``/results/*``, ``/halo/*``, ``axes/m_phi_gev``) is sliced accordingly. Every
kept cell is byte-identical to the parent cube -- nothing is recomputed,
re-encoded or re-rounded. Everything without a lambda axis (detector inputs,
efficiency curves, reference curves, halo axes, root provenance attrs) is
copied verbatim.

The output keeps the parent's layout conventions (assemble_release.py):
gzip-4 + shuffle with one-hypothesis-plane chunks on the cubes, HDF5 dimension
scales with labels and the flat ``axes`` attribute, ``n_finite`` and
``tags_json`` on ``axes/lambda_m`` (tags whose value is dropped are removed).

Subset provenance is recorded on the file itself:

  version_tag             the new tag (--version-tag)
  subset_of_version_tag   the parent's version_tag
  subset_parent_file      basename of the parent file
  subset_parent_sha256    SHA256 of the parent file
  subset_date             the subset date (--subset-date)
  subset_lambda_kept_m    the finite lambda values kept (m)
  subset_note             what happened, in one sentence

All other root attrs (created, git_commit, seed, fidelity, input hashes, ...)
are carried over unchanged: they describe how the physics was computed, and the
physics is unchanged.

The v5 -> v6 release job::

    python scripts/subset_release_lambda.py \
        --in  release/luhdm_datarelease_v5.h5 \
        --out release/luhdm_datarelease_v6.h5 \
        --keep 2e-5 2e-4 2e-3 \
        --version-tag v6.0-night-m0p356mg-bcap10cm-lam4 \
        --subset-date 2026-08-12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# HDF5 dimension-scale bookkeeping attrs: managed by the dims API, never copied.
_SCALE_ATTRS = {"CLASS", "NAME", "REFERENCE_LIST", "DIMENSION_LIST",
                "DIMENSION_LABELS"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dims_of(ds):
    """Axis names of a dataset from its flat ``axes`` attr, or None."""
    axes = ds.attrs.get("axes")
    if axes is None:
        return None
    if isinstance(axes, bytes):
        axes = axes.decode("utf-8")
    return [s.strip() for s in str(axes).split(",")]


def kept_lambda_indices(lam_axis, keep):
    """Indices into the parent lambda axis: kept finite ascending, then inf."""
    idx = []
    for v in sorted(float(k) for k in keep):
        hits = np.where(np.isclose(lam_axis, v, rtol=1e-9, atol=0.0))[0]
        if hits.size != 1:
            sys.exit(f"FATAL: lambda {v!r} m matches {hits.size} axis entries "
                     f"(need exactly 1); the finite axis spans "
                     f"{lam_axis[np.isfinite(lam_axis)].min():g}.."
                     f"{lam_axis[np.isfinite(lam_axis)].max():g} m")
        idx.append(int(hits[0]))
    inf_hits = np.where(np.isinf(lam_axis))[0]
    if inf_hits.size != 1:
        sys.exit(f"FATAL: expected exactly one massless (inf) sentinel on the "
                 f"lambda axis; found {inf_hits.size}")
    idx.append(int(inf_hits[0]))
    return idx


def _copy_attrs(src, dst, skip=()):
    for k, v in src.attrs.items():
        if k in _SCALE_ATTRS or k in skip:
            continue
        dst.attrs[k] = v


def _subset_chunks(src_chunks, lam_pos, new_len):
    if src_chunks is None:
        return None
    ch = list(src_chunks)
    ch[lam_pos] = min(ch[lam_pos], new_len)
    return tuple(ch)


def subset_release(in_path, out_path, keep, version_tag, subset_date):
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.resolve() == in_path.resolve():
        sys.exit("FATAL: refusing to overwrite the parent file")
    parent_sha = sha256_file(in_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    with h5py.File(in_path, "r") as src, h5py.File(tmp, "w") as dst:
        lam_axis = src["axes/lambda_m"][:]
        idx = kept_lambda_indices(lam_axis, keep)
        n_finite_new = len(idx) - 1
        print(f"parent lambda axis: {lam_axis.size} entries "
              f"({int(src['axes/lambda_m'].attrs['n_finite'])} finite); "
              f"keeping indices {idx} -> "
              f"{[float(lam_axis[i]) for i in idx]}")

        # tags: keep only tags whose value survives on the new axis
        old_tags = json.loads(src["axes/lambda_m"].attrs["tags_json"])
        new_tags = {t: v for t, v in old_tags.items()
                    if any(np.isclose(v, lam_axis[i], rtol=1e-9, atol=0.0)
                           for i in idx[:-1])}
        print(f"lambda tags: {sorted(old_tags)} -> {sorted(new_tags)} "
              f"(+ the massless sentinel, which is a value, not a tag)")

        scale_paths = []      # datasets that are dimension scales in the parent
        attachments = []      # (dset path, dim index, scale path, label)

        def walk(name, obj):
            if isinstance(obj, h5py.Group):
                g = dst.create_group(name)
                _copy_attrs(obj, g)
                return
            # -- datasets ---------------------------------------------------
            path = obj.name.lstrip("/")
            dims = _dims_of(obj)
            if path == "axes/lambda_m":
                data = obj[idx]
                lam_pos = None
            elif path == "axes/m_phi_gev":
                data = obj[idx]          # parallel to lambda_m by construction
                lam_pos = None
            elif dims is not None and "lambda_m" in dims:
                lam_pos = dims.index("lambda_m")
                data = np.take(obj[()], idx, axis=lam_pos)
            else:
                data = obj[()]
                lam_pos = None

            kw = {}
            if obj.compression is not None:
                kw = dict(compression=obj.compression,
                          compression_opts=obj.compression_opts,
                          shuffle=obj.shuffle,
                          chunks=(obj.chunks if lam_pos is None else
                                  _subset_chunks(obj.chunks, lam_pos,
                                                 len(idx))))
            d = dst.create_dataset(path, data=data, **kw)
            _copy_attrs(obj, d,
                        skip=("n_finite", "tags_json")
                        if path == "axes/lambda_m" else ())
            if path == "axes/lambda_m":
                d.attrs["n_finite"] = int(n_finite_new)
                d.attrs["tags_json"] = json.dumps(new_tags)

            # record the parent's dimension-scale wiring to reproduce it
            if obj.attrs.get("CLASS") == b"DIMENSION_SCALE":
                scale_paths.append(path)
            for i, dim in enumerate(obj.dims):
                for scale in dim.values():
                    attachments.append((path, i, scale.name.lstrip("/"),
                                        dim.label))

        src.visititems(walk)

        for p in scale_paths:
            dst[p].make_scale(p.rsplit("/", 1)[-1])
        for dpath, i, spath, label in attachments:
            dst[dpath].dims[i].attach_scale(dst[spath])
            dst[dpath].dims[i].label = label

        # -- root attrs: everything carried over, then the subset provenance --
        _copy_attrs(src, dst)
        parent_tag = src.attrs["version_tag"]
        a = dst.attrs
        a["version_tag"] = version_tag
        a["subset_of_version_tag"] = parent_tag
        a["subset_parent_file"] = in_path.name
        a["subset_parent_sha256"] = parent_sha
        a["subset_date"] = subset_date
        a["subset_lambda_kept_m"] = np.asarray(sorted(float(k) for k in keep),
                                               dtype=np.float64)
        a["subset_note"] = (
            f"axis-subset of {parent_tag}: lambda axis reduced to "
            f"{n_finite_new} finite mediator ranges "
            f"({', '.join(sorted(new_tags, key=new_tags.get))}) plus the "
            f"massless sentinel. Same physics, no recompute: every kept cell "
            f"is bit-identical to the parent cube.")

    os.replace(tmp, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB; parent "
          f"{in_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="parent release HDF5 (read-only)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="output subset HDF5")
    ap.add_argument("--keep", nargs="+", type=float, required=True,
                    help="finite lambda values to keep, in metres (exact axis "
                         "members, rtol 1e-9); the massless sentinel is always "
                         "kept")
    ap.add_argument("--version-tag", required=True,
                    help="version_tag of the subset file")
    ap.add_argument("--subset-date", required=True,
                    help="date recorded as the subset_date attr (YYYY-MM-DD)")
    args = ap.parse_args()
    subset_release(args.in_path, args.out_path, args.keep, args.version_tag,
                   args.subset_date)


if __name__ == "__main__":
    main()
