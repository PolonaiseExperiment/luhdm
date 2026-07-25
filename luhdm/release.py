"""Reader for the POLONAISE UHDM data release (HDF5).

The release is a single self-describing HDF5 file
(``release/luhdm_datarelease_v2.h5``) holding the analysis as a matrix over
(sensor mode, coupling ``alpha_n``, dark-matter mass, mediator range ``lambda``).
It is produced by ``scripts/build_release.py`` (the per-lambda shard builder) and
``scripts/assemble_release.py`` (shards -> HDF5); this module only *reads* it.

The file is intentionally plain HDF5 with dimension scales and rich per-dataset
``units``/``description`` attributes, so it can be opened by any HDF5 tool
(``h5py``, ``h5ls``, ``h5dump``, MATLAB, ...) with no luhdm code. This module is
the convenience layer: eager axis loading, exact/nearest index resolution,
hyperslab plane reads, and the shared best-mass criterion so that notebooks 01
and 04 cannot drift from each other.

Datasets use axis order ``(mode, alpha, mass, lambda)``. The lambda axis is the
finite ranges in ascending order followed by ``np.inf`` (the massless / analytic
slice) as the last element, for which ``m_phi_gev`` is exactly ``0.0``. Per-mode
cubes are ``(3, n_alpha, n_mass, n_lambda)``; ``n_transit`` is mode-less
``(n_alpha, n_mass, n_lambda)``.

``h5py`` is imported here (not in ``luhdm.__init__``) so importing the rest of
the package stays dependency-light.

Quickstart::

    from luhdm import release
    rel = release.open_release()
    ext = rel.mass_plane("extremeness", mode=1, lam="200um")   # (n_alpha, n_mass)
    m_best, im = rel.best_mass(mode=1)
    rel.close()
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from luhdm import limits

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "release" / "luhdm_datarelease_v2.h5"

FORMAT_VERSION = 1

# Canonical mediator-range tags -> lambda in metres. Order matters elsewhere;
# 'massless' -> inf is the analytic slice appended last on the lambda axis.
TAGS = {
    "2m": 2.0,
    "2cm": 2e-2,
    "2mm": 2e-3,
    "200um": 2e-4,
    "20um": 2e-5,
    "10um": 1e-5,
    "2um": 2e-6,
    "massless": np.inf,
}

# Canonical ordering used throughout the release / notebooks.
TAG_ORDER = ["2m", "2cm", "2mm", "200um", "20um", "10um", "2um"]
# Speed-distribution tags: standard halo model + the seven finite ranges.
SPEED_TAGS = ("shm",) + tuple(TAG_ORDER)
# Raw-spectrum tags: the seven finite ranges + the massless slice.
RAW_TAGS = tuple(TAG_ORDER) + ("massless",)

_GROUP_H5 = {"atm": "atm", "noatm": "noatm", "halo": "halo"}
_PER_MODE = ("extremeness", "mu", "status")


# --------------------------------------------------------------------------- #
# Pure best-mass criterion (shared by notebooks 01 + 04; no I/O).
# --------------------------------------------------------------------------- #
def best_mass_index(p_finite, lambda_finite, alpha_n, confidence=0.95):
    """Index of the "best" dark-matter mass in an extremeness cube.

    Replicates notebook 01's best-mass criterion (cell "best mass = widest
    95% CL exclusion ..."): the best mass has the largest excluded log-alpha
    area integrated over log-lambda, *restricted* to the masses that reach the
    globally shortest excluded mediator length.

    Parameters
    ----------
    p_finite : ndarray, shape ``(n_alpha, n_mass, n_finite)``
        Optimum-interval extremeness over the finite lambda axis, indexed
        ``p[alpha, mass, lambda]`` (the per-mode slice of the release cube with
        the trailing massless column dropped).
    lambda_finite : ndarray, shape ``(n_finite,)``
        Finite mediator ranges in metres, ascending.
    alpha_n : ndarray, shape ``(n_alpha,)``
        Coupling axis (log-spaced); only its mean log-step enters, matching the
        notebook's ``np.diff(np.log10(alphas)).mean()`` weight.
    confidence : float
        Exclusion level (default 0.95).

    Returns
    -------
    int
        Index into the mass axis.
    """
    p = np.asarray(p_finite, dtype=float)
    lam = np.asarray(lambda_finite, dtype=float)
    alpha = np.asarray(alpha_n, dtype=float)
    if p.ndim != 3:
        raise ValueError(f"p_finite must be 3-D (alpha, mass, lambda); got {p.shape}")
    n_alpha, n_mass, n_finite = p.shape
    dloga = np.mean(np.diff(np.log10(alpha)))
    # W[il, im] = (# alphas excluded at this mass/lambda) * mean d log alpha.
    # (p >= confidence).sum over the alpha axis -> (mass, lambda); transpose to
    # (lambda, mass) to integrate over lambda along axis 0.
    W = (p >= confidence).sum(axis=0).T * dloga          # (n_finite, n_mass)
    loglam = np.log10(lam)
    area = np.trapezoid(W, loglam, axis=0)               # (n_mass,)
    shortest = np.full(n_mass, np.inf)
    for im in range(n_mass):
        reached = np.where(W[:, im] > 0)[0]
        if reached.size:
            shortest[im] = lam[reached].min()
    cand = np.where(shortest == shortest.min())[0]
    return int(cand[np.argmax(area[cand])])


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _py(v):
    """Normalise an HDF5 attribute value to a plain Python object."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class _AxisSet:
    """A namespace of axis arrays for one group (atm / noatm / halo).

    ``mass_gev`` is that group's mass axis; ``alpha_n`` its coupling axis.
    ``lambda_m`` / ``m_phi_gev`` are the shared lambda axis (finite ascending +
    inf last); ``lambda_finite`` drops the trailing inf; ``n_finite`` is the
    number of finite lambda points.
    """

    def __init__(self, mass_gev, alpha_n, lambda_m, m_phi_gev, n_finite):
        self.mass_gev = np.asarray(mass_gev)
        self.alpha_n = np.asarray(alpha_n)
        self.lambda_m = None if lambda_m is None else np.asarray(lambda_m)
        self.m_phi_gev = None if m_phi_gev is None else np.asarray(m_phi_gev)
        self.n_finite = n_finite

    @property
    def lambda_finite(self):
        return self.lambda_m[:-1]


# --------------------------------------------------------------------------- #
# Open
# --------------------------------------------------------------------------- #
def open_release(path=None):
    """Open the data release, returning a :class:`Release`.

    ``path`` defaults to :data:`DEFAULT_PATH`. A missing file raises
    ``FileNotFoundError`` pointing at the regeneration instructions; a
    ``file_format`` / ``version`` mismatch raises ``ValueError`` naming both
    the file's and the loader's expected version.
    """
    path = DEFAULT_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data release not found: {path}\n"
            f"See release/README.md and regenerate it with "
            f"scripts/build_release.py (build shards) + "
            f"scripts/assemble_release.py (assemble the HDF5)."
        )
    f = h5py.File(path, "r")
    ff = _py(f.attrs.get("file_format"))
    ver = f.attrs.get("version")
    ver_i = None if ver is None else int(ver)
    if ff != "luhdm-datarelease" or ver_i != FORMAT_VERSION:
        f.close()
        raise ValueError(
            f"unrecognised data release: file_format={ff!r}, version={ver_i!r}; "
            f"this loader (luhdm.release) expects "
            f"file_format='luhdm-datarelease', version={FORMAT_VERSION}."
        )
    return Release(f, path)


class Release:
    """Read-only handle on one data-release HDF5 file.

    Usable as a context manager (``with open_release() as rel: ...``) or opened
    plainly in a notebook (call :meth:`close` when done). Axes are read eagerly
    into the ``axes`` / ``axes_noatm`` / ``axes_halo`` namespaces; cube reads are
    lazy hyperslabs.
    """

    def __init__(self, f, path=None):
        self._file = f
        self.path = None if path is None else Path(path)
        self.attrs = {k: _py(v) for k, v in f.attrs.items()}
        self._best_mass_cache = {}
        self._read_axes()

    # -- lifecycle -------------------------------------------------------- #
    def _read_axes(self):
        ax = self._file["axes"]
        alpha = ax["alpha_n"][:]
        lam = ax["lambda_m"][:]
        mphi = ax["m_phi_gev"][:]
        n_finite = int(ax["lambda_m"].attrs["n_finite"]) if "n_finite" in \
            ax["lambda_m"].attrs else int(np.count_nonzero(np.isfinite(lam)))
        self._mode = ax["mode"][:]
        self.axes = _AxisSet(ax["mass_gev"][:], alpha, lam, mphi, n_finite)
        self.axes_noatm = _AxisSet(ax["mass_noatm_gev"][:], alpha, lam, mphi, n_finite)
        self.axes_halo = _AxisSet(
            ax["mass_halo_gev"][:], ax["alpha_halo_n"][:], lam, mphi, n_finite)

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return f"<Release {self.path} ({self.version_tag})>"

    @property
    def version_tag(self):
        """Human-readable cube version, e.g. 'v1.0' or 'v2.0-bcap10cm'.

        Set at assembly time (``assemble_release.py --version-tag``). Notebooks
        stamp it onto every figure so a figure always names the cube it came
        from. Falls back to the numeric format version for pre-tag files.
        """
        return str(self.attrs.get("version_tag")
                   or f"v{self.attrs.get('version', FORMAT_VERSION)}")

    @property
    def b_constrained_max(self):
        """Impact-parameter cap in metres, or None if the cube is uncapped.

        When set, the b-integral's outer limit is min(b_constrained_max,
        b_max(q)) in both dsigma/dq and the geometric transit reach.
        """
        v = self.attrs.get("b_constrained_max_m")
        return None if v is None or not np.isfinite(v) else float(v)

    # -- internal resolvers ----------------------------------------------- #
    def _axis_for(self, group):
        if group == "atm":
            return self.axes
        if group == "noatm":
            return self.axes_noatm
        if group == "halo":
            return self.axes_halo
        raise ValueError(f"unknown group {group!r}; expected 'atm', 'noatm' or 'halo'")

    def _h5group(self, group):
        if group not in _GROUP_H5:
            raise ValueError(
                f"unknown group {group!r}; expected 'atm', 'noatm' or 'halo'")
        return self._file[_GROUP_H5[group]]

    def _mode_index(self, mode):
        if mode is None:
            raise ValueError("mode is required for this quantity (per-mode cube)")
        idx = np.where(self._mode == int(mode))[0]
        if idx.size == 0:
            raise ValueError(f"mode {mode!r} not in {self._mode.tolist()}")
        return int(idx[0])

    @staticmethod
    def _lam_to_float(lam):
        if isinstance(lam, str):
            if lam not in TAGS:
                raise KeyError(
                    f"unknown lambda tag {lam!r}; known tags: {sorted(TAGS)}")
            return TAGS[lam]
        return float(lam)

    # -- index resolution ------------------------------------------------- #
    def at_lambda(self, lam, group="atm"):
        """Exact lambda-axis index (rtol 1e-9). Miss -> KeyError with 3 nearest.

        Accepts a tag string (incl. ``'massless'``), a float in metres, or
        ``np.inf``. No silent snapping: a value not on the axis raises.
        """
        axis = self._axis_for(group).lambda_m
        target = self._lam_to_float(lam)
        if np.isinf(target):
            matches = np.isinf(axis) & (np.sign(axis) == np.sign(target))
        else:
            matches = np.isclose(axis, target, rtol=1e-9, atol=0.0)
        idx = np.where(matches)[0]
        if idx.size:
            return int(idx[0])
        loga = np.log10(target) if target > 0 else -np.inf
        axis_log = np.where(np.isfinite(axis) & (axis > 0), np.log10(axis), np.inf)
        dist = np.abs(axis_log - loga)
        order = np.argsort(dist)[:3]
        nearest = [float(x) for x in axis[order]]
        raise KeyError(
            f"lambda {target!r} m is not an exact axis value (rtol 1e-9); "
            f"3 nearest axis values (m): {nearest}")

    def at_mass(self, mass, group="atm"):
        """Nearest mass-axis index in log10 (group selects the mass axis)."""
        axis = self._axis_for(group).mass_gev
        return int(np.argmin(np.abs(np.log10(axis) - np.log10(float(mass)))))

    def at_alpha(self, alpha, group="atm"):
        """Nearest coupling-axis index in log10 (group selects the alpha axis)."""
        axis = self._axis_for(group).alpha_n
        return int(np.argmin(np.abs(np.log10(axis) - np.log10(float(alpha)))))

    # -- plane reads ------------------------------------------------------ #
    def mass_plane(self, quantity, mode=None, lam="200um", group="atm"):
        """A (n_alpha, n_mass) plane at fixed lambda.

        ``quantity`` in ``{extremeness, mu, n_transit, status}`` for atm/noatm
        (``extremeness``/``mu``/``status`` need ``mode``; ``n_transit`` is
        mode-less), or ``{n_transit, bmax}`` for ``group='halo'`` (``mode`` must
        be ``None``).
        """
        if group == "halo":
            if mode is not None:
                raise ValueError("halo quantities are mode-less; pass mode=None")
            if quantity not in ("n_transit", "bmax"):
                raise ValueError(
                    f"halo quantity must be 'n_transit' or 'bmax'; got {quantity!r}")
            il = self.at_lambda(lam, "halo")
            return self._h5group("halo")[quantity][:, :, il]

        if quantity not in ("extremeness", "mu", "n_transit", "status"):
            raise ValueError(
                f"{group} quantity must be one of extremeness/mu/n_transit/status; "
                f"got {quantity!r}")
        il = self.at_lambda(lam, group)
        ds = self._h5group(group)[quantity]
        if quantity == "n_transit":
            if mode is not None:
                raise ValueError("n_transit is mode-less; pass mode=None")
            return ds[:, :, il]
        imode = self._mode_index(mode)
        return ds[imode, :, :, il]

    def lambda_plane(self, quantity, mode, mass=None, group="atm"):
        """A (n_alpha, n_finite) plane at fixed mass over the finite lambdas.

        ``mass=None`` uses :meth:`best_mass` for this mode/group. Only atm/noatm
        groups; the trailing massless lambda column is excluded.
        """
        if group not in ("atm", "noatm"):
            raise ValueError(
                f"lambda_plane is defined for 'atm'/'noatm'; got {group!r}")
        if quantity not in ("extremeness", "mu", "n_transit", "status"):
            raise ValueError(
                f"quantity must be one of extremeness/mu/n_transit/status; "
                f"got {quantity!r}")
        axset = self._axis_for(group)
        nfin = axset.n_finite
        im = self.best_mass(mode, group)[1] if mass is None \
            else self.at_mass(mass, group)
        ds = self._h5group(group)[quantity]
        if quantity == "n_transit":
            return ds[:, im, :nfin]
        imode = self._mode_index(mode)
        return ds[imode, :, im, :nfin]

    def composite(self, quantity="extremeness", lam="massless", group="noatm"):
        """Per-mode maximum of a plane: ``np.maximum.reduce`` over the 3 modes."""
        if quantity not in _PER_MODE:
            raise ValueError(
                f"composite needs a per-mode quantity {_PER_MODE}; got {quantity!r}")
        il = self.at_lambda(lam, group)
        ds = self._h5group(group)[quantity]
        planes = [ds[k, :, :, il] for k in range(ds.shape[0])]
        return np.maximum.reduce(planes)

    # -- derived ---------------------------------------------------------- #
    def best_mass(self, mode, group="atm", confidence=0.95):
        """(mass_gev, im): the mode's best mass via :func:`best_mass_index`.

        Memoised per ``(mode, group, confidence)``. Loads the mode's
        finite-lambda extremeness cube once and delegates to the shared pure
        criterion.
        """
        if group not in ("atm", "noatm"):
            raise ValueError(
                f"best_mass is defined for 'atm'/'noatm'; got {group!r}")
        key = (int(mode), group, float(confidence))
        if key in self._best_mass_cache:
            return self._best_mass_cache[key]
        axset = self._axis_for(group)
        imode = self._mode_index(mode)
        nfin = axset.n_finite
        p_finite = self._h5group(group)["extremeness"][imode, :, :, :nfin]
        im = best_mass_index(p_finite, axset.lambda_finite, axset.alpha_n,
                             confidence)
        result = (float(axset.mass_gev[im]), im)
        self._best_mass_cache[key] = result
        return result

    def excluded_alpha_band(self, mass, lam, mode=1, group="atm", confidence=0.95):
        """(lo, hi) coupling edges excluded at fixed (mass, lambda, mode).

        Delegates to :func:`luhdm.limits.excluded_band` on the 1-D
        ``p(alpha)`` column; ``(nan, nan)`` if nothing is excluded.
        """
        axset = self._axis_for(group)
        im = self.at_mass(mass, group)
        il = self.at_lambda(lam, group)
        imode = self._mode_index(mode)
        column = self._h5group(group)["extremeness"][imode, :, im, il]
        return limits.excluded_band(axset.alpha_n, column, level=confidence)

    def cell(self, mass, alpha, lam, mode=1, group="atm"):
        """Full dump of one (mass, alpha, lambda, mode) cell as a dict.

        Includes the atm/noatm values at the resolved cell plus the halo values
        (``bmax``, ``n_transit_halo``) from the nearest halo cell at the same
        lambda.
        """
        axset = self._axis_for(group)
        ia = self.at_alpha(alpha, group)
        im = self.at_mass(mass, group)
        il = self.at_lambda(lam, group)
        imode = self._mode_index(mode)
        g = self._h5group(group)
        # nearest halo cell at the same lambda index (halo shares the lambda axis)
        ia_h = self.at_alpha(alpha, "halo")
        im_h = self.at_mass(mass, "halo")
        halo = self._h5group("halo")
        return {
            "mass_gev": float(axset.mass_gev[im]),
            "alpha_n": float(axset.alpha_n[ia]),
            "lambda_m": float(axset.lambda_m[il]),
            "indices": (ia, im, il),
            "extremeness": float(g["extremeness"][imode, ia, im, il]),
            "mu": float(g["mu"][imode, ia, im, il]),
            "n_transit": float(g["n_transit"][ia, im, il]),
            "status": int(g["status"][imode, ia, im, il]),
            "bmax": float(halo["bmax"][ia_h, im_h, il]),
            "n_transit_halo": float(halo["n_transit"][ia_h, im_h, il]),
        }

    # -- detector / reference curves ------------------------------------- #
    def events(self, mode):
        """Observed impulse candidates for a mode (GeV)."""
        return self._file["detector"][f"events_mode{int(mode)}"][:]

    def all_blips(self, mode):
        """All reconstructed blip momenta for a mode (eV)."""
        return self._file["detector"][f"all_blips_mode{int(mode)}"][:]

    def efficiency_curve(self, mode, df=3):
        """(q_gev, efficiency) for a mode and dof hypothesis (df in {2, 3})."""
        det = self._file["detector"]
        return det[f"q_gev_{int(mode)}"][:], det[f"eff_{int(mode)}_df{int(df)}"][:]

    def speed_dist(self, tag):
        """(v, f_v) arrival-speed distribution for ``tag`` in SPEED_TAGS."""
        if tag not in SPEED_TAGS:
            raise ValueError(f"speed_dist tag must be one of {SPEED_TAGS}; got {tag!r}")
        rc = self._file["reference_curves"]
        return rc["v"][:], rc[f"fv_{tag}"][:]

    def survival_fraction(self, tag):
        """Attenuation survival fraction stored on the ``fv_<tag>`` dataset."""
        if tag not in SPEED_TAGS:
            raise ValueError(
                f"survival_fraction tag must be one of {SPEED_TAGS}; got {tag!r}")
        return float(self._file["reference_curves"][f"fv_{tag}"].attrs["survival_fraction"])

    def raw_spectrum(self, tag):
        """(q_gev, dR/dq) raw spectrum for ``tag`` in RAW_TAGS (incl. 'massless')."""
        if tag not in RAW_TAGS:
            raise ValueError(f"raw_spectrum tag must be one of {RAW_TAGS}; got {tag!r}")
        rc = self._file["reference_curves"]
        return rc["q_gev"][:], rc[f"drdq_{tag}"][:]

    # -- introspection ---------------------------------------------------- #
    def tree(self):
        """An indented listing of groups/datasets with shape, dtype and units."""
        lines = ["/"]

        def visit(name, obj):
            depth = name.count("/") + 1
            indent = "  " * depth
            base = name.rsplit("/", 1)[-1]
            if isinstance(obj, h5py.Dataset):
                units = _py(obj.attrs.get("units", ""))
                tail = f"  [{units}]" if units else ""
                lines.append(f"{indent}{base}  {tuple(obj.shape)} {obj.dtype}{tail}")
            else:
                lines.append(f"{indent}{base}/")

        self._file.visititems(visit)
        return "\n".join(lines)
