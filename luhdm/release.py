"""Reader for the POLONAISE UHDM data release (HDF5).

A release file is a self-describing HDF5 cube holding the analysis as a matrix
over (sensor mode, coupling ``alpha_n``, dark-matter mass, mediator range
``lambda``) for one or more ``(f_dm, atmosphere)`` hypotheses. It is produced by
``scripts/build_release.py`` (the per-lambda shard builder) and
``scripts/assemble_release.py`` (shards -> HDF5); this module only *reads* it.

The **public** release is the v9.1 pair, one hypothesis per file:
``release/luhdm_datarelease_v9p1_A_f1_atm.h5`` (f_DM = 1, attenuated) and
``release/luhdm_datarelease_v9p1_B_f0p1_noatm.h5`` (f_DM = 0.1, bare halo). Pass
the one you want to :func:`open_release`; note that a single-hypothesis file may
not carry ``attrs['f_dm_default']``, so pass ``f_dm=`` explicitly rather than
relying on the fallback. :data:`DEFAULT_PATH` is file A of that pair, so a bare
``open_release()`` (and the ``--release`` default of the paper-figure scripts)
lands on the published attenuated cube.

Neither the release files nor this reader apply the release's mass cut
``m_cut``: the v9 cross section is uncapped, so the surfaces report exclusion
above the mass where the halo stops delivering transits through the apparatus,
and the cut that closes the region is a root attribute (``m_cut_*``) the caller
applies. See section 5.4 of ``release/README.md``.

A cube also records the *projection kernel* its cross sections were built with
(root attribute ``projection_kernel``; v9 is ``isotropic-folded``, pre-flag
files carry no attribute and were built ``planar-signed``). Recomputing a cell
under the other kernel is a silent physics error, so use :meth:`Release.make_xsec`
rather than calling :func:`luhdm.rate.make_xsec` directly: it threads the file's
own kernel and impact-parameter cap into the handle. See section 5.5 of
``release/README.md``.

The file is intentionally plain HDF5 with dimension scales and rich per-dataset
``units``/``description`` attributes, so it can be opened by any HDF5 tool
(``h5py``, ``h5ls``, ``h5dump``, MATLAB, ...) with no luhdm code. This module is
the convenience layer: eager axis loading, exact/nearest index resolution,
hyperslab plane reads, and the shared best-mass criterion so that notebooks 01
and 04 cannot drift from each other.

Two on-disk layouts are readable, detected automatically:

``axes`` (file ``version`` 2, the release layout)
    One ``/results/<quantity>`` array per quantity, shaped
    ``(f_dm, atmosphere, mode, alpha, mass, lambda)``, so every element
    explicitly carries its hypothesis. ``axes/f_dm`` is ``[0.1, 1.0]`` and
    ``axes/atmosphere`` is ``[1, 0]`` (1 = attenuation applied). ``n_transit``
    drops the mode axis: ``(f_dm, atmosphere, alpha, mass, lambda)``.

``groups`` (file ``version`` 1, the v3 layout)
    ``/atm`` and ``/noatm`` groups holding ``(mode, alpha, mass, lambda)``
    cubes, with the f_DM = 1 surface in parallel ``*_f1`` datasets and a mass
    axis per pass.

The lambda axis is the finite ranges in ascending order followed by ``np.inf``
(the massless / analytic slice) as the last element, for which ``m_phi_gev`` is
exactly ``0.0``.

``h5py`` is imported here (not in ``luhdm.__init__``) so importing the rest of
the package stays dependency-light.

Cubes built by the dual-f_DM pipeline carry two exclusion surfaces: the baseline
dark-matter fraction f_DM = 0.1 and f_DM = 1.0. Because f_DM is a pure flux
normalisation, both come from one campaign. Every accessor takes ``f_dm=``
(default 0.1) and ``atmosphere=`` (default True), and the historical
``group='atm'/'noatm'`` spelling of the atmosphere choice keeps working, so code
written before either axis existed reads exactly what it always did.

Quickstart::

    from luhdm import release
    rel = release.open_release()
    ext = rel.mass_plane("extremeness", mode=1, lam="200um")   # (n_alpha, n_mass)
    ext1 = rel.mass_plane("extremeness", mode=1, lam="200um", f_dm=1.0)
    bare = rel.mass_plane("extremeness", mode=1, lam="200um", atmosphere=False)
    m_best, im = rel.best_mass(mode=1)
    rel.close()
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from luhdm import limits

# File A of the public pair (f_DM = 1, attenuated): see the module docstring.
# Also the ``--release`` default of every scripts/paper_fig_*.py.
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "release" / "luhdm_datarelease_v9p1_A_f1_atm.h5"

# Projection kernel of cubes that predate the ``projection_kernel`` attribute:
# they name no kernel and were in fact built with exactly this convention. Same
# fallback, same spelling, as scripts/refine_contours.py.
KERNEL_PRE_FLAG = "planar-signed"

FORMAT_VERSION = 1            # v3 group layout (/atm, /noatm)
FORMAT_VERSION_AXES = 2       # axis layout (/results over f_dm x atmosphere)
SUPPORTED_VERSIONS = (FORMAT_VERSION, FORMAT_VERSION_AXES)

# Canonical mediator-range tags -> lambda in metres. Order matters elsewhere;
# 'massless' -> inf is the analytic slice appended last on the lambda axis.
TAGS = {
    "2m": 2.0,
    "20cm": 0.2,
    "2cm": 2e-2,
    "2mm": 2e-3,
    "200um": 2e-4,
    "20um": 2e-5,
    "10um": 1e-5,
    "2um": 2e-6,
    "massless": np.inf,
}

# Canonical ordering used throughout the release / notebooks.
TAG_ORDER = ["2m", "20cm", "2cm", "2mm", "200um", "20um", "10um", "2um"]
# Speed-distribution tags: standard halo model + the eight finite ranges.
SPEED_TAGS = ("shm",) + tuple(TAG_ORDER)
# Raw-spectrum tags: the eight finite ranges + the massless slice.
RAW_TAGS = tuple(TAG_ORDER) + ("massless",)

_GROUP_H5 = {"atm": "atm", "noatm": "noatm", "halo": "halo"}
_PER_MODE = ("extremeness", "mu", "status")
_QUANTITIES = ("extremeness", "mu", "n_transit", "status")

# Atmosphere hypothesis <-> the historical group name. The atm/noatm split was
# always the presence or absence of the attenuation ODE; in the axis layout it
# is an explicit axis (``axes/atmosphere``, 1 = attenuation on) instead of two
# groups, and ``group=`` keeps working as the name for the same choice.
_ATM_GROUP = {True: "atm", False: "noatm"}

# Dark-matter fraction hypotheses carried by the cube. f_DM is a pure flux
# normalisation (same attenuation ODE, same dR/dq shape, mu scales linearly), so
# one campaign produces both surfaces: the baseline f_DM = 0.1 in the unsuffixed
# datasets and f_DM = 1.0 in the parallel ``*_f1`` ones. Every accessor takes
# ``f_dm=`` and defaults to the baseline, so pre-dual-f code is unaffected.
F_DM_DEFAULT = 0.1
F_DM_VALUES = (0.1, 1.0)
_F_DM_SUFFIX = {0.1: "", 1.0: "_f1"}


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
    # (lambda, mass) to integrate over lambda along axis 0. The level is taken
    # at the file's float32 storage precision so a builder value exactly at the
    # level survives the narrowing.
    W = (p >= np.float32(confidence)).sum(axis=0).T * dloga  # (n_finite, n_mass)
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
    if ff != "luhdm-datarelease" or ver_i not in SUPPORTED_VERSIONS:
        f.close()
        raise ValueError(
            f"unrecognised data release: file_format={ff!r}, version={ver_i!r}; "
            f"this loader (luhdm.release) expects "
            f"file_format='luhdm-datarelease', version in "
            f"{list(SUPPORTED_VERSIONS)}."
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
        # layout detection: the axis layout puts everything in /results over an
        # explicit (f_dm, atmosphere) pair of axes; the v3 layout splits the
        # atmosphere hypothesis into /atm and /noatm groups with its own mass
        # axis each. Both are readable through the same accessors.
        self.layout = "axes" if "results" in self._file else "groups"
        if self.layout == "axes":
            self._f_dm_axis = np.asarray(ax["f_dm"][:], dtype=float)
            self._atmosphere_axis = np.asarray(ax["atmosphere"][:], dtype=int)
            mass_noatm = ax["mass_gev"][:]        # one shared mass axis
        else:
            self._f_dm_axis = None
            self._atmosphere_axis = None
            mass_noatm = ax["mass_noatm_gev"][:]
        self.axes = _AxisSet(ax["mass_gev"][:], alpha, lam, mphi, n_finite)
        self.axes_noatm = _AxisSet(mass_noatm, alpha, lam, mphi, n_finite)
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

    @property
    def projection_kernel(self):
        """Projected-dsigma/dq convention this cube's surfaces were built with.

        ``'isotropic-folded'`` (v9 onward: the absolute one-axis projection under
        isotropic arrivals, coefficient 8 pi/3 and x^3 shell fraction) or
        ``'planar-signed'`` (the historical signed-projection kernel, coefficient
        2 pi and arcsine shell fraction). Files that predate the attribute name no
        kernel and were built planar-signed, so that is the fallback --
        :data:`KERNEL_PRE_FLAG`, the same convention ``scripts/refine_contours.py``
        applies when it dispatches a cube's kernel into its own recomputation.
        """
        return str(self.attrs.get("projection_kernel", KERNEL_PRE_FLAG))

    @property
    def v_earth_km_s(self):
        """Halo frame this cube's surfaces were built in, in km/s.

        ``0.0`` is the Galactic rest frame: the truncated Maxwellian of
        :func:`luhdm.halo.standard_halo_model` with no Earth motion, which is
        what every cube predating this attribute carries. A non-zero value is
        the lab-frame distribution (Lewin & Smith 1996; Monteiro 2020, Tseng
        2025), whose support runs to v_esc + v_Earth rather than v_esc.

        Recomputing a cell of such a cube requires the SAME frame in
        :mod:`luhdm.config` -- set the ``LUHDM_V_EARTH`` environment variable to
        this value before importing luhdm. Nothing in the recomputed numbers
        announces the mismatch, so ``scripts/refine_contours.py`` and
        ``scripts/verify_release.py`` both hard-stop on it.
        """
        return float(self.attrs.get("v_earth_km_s", 0.0))

    def make_xsec(self, lamb, **kw):
        """A :func:`luhdm.rate.make_xsec` handle under *this cube's* conventions.

        The one supported way to recompute a cell of the release: it fills in the
        file's :attr:`projection_kernel` and :attr:`b_constrained_max` so a
        recomputed dsigma/dq, dR/dq or mu is comparable with the stored surface
        instead of silently mixing two projection conventions (the module-level
        defaults in :mod:`luhdm.rate` are the *historical* ones, which pre-v9
        cubes use and v9 does not).

        ``lamb`` is the mediator range in metres, or ``None`` for the massless
        slice. Any other :func:`luhdm.rate.make_xsec` keyword passes through;
        naming ``projection_kernel`` or ``b_constrained_max`` explicitly
        overrides the file's value, which is a deliberate act, not an accident.
        """
        # imported lazily: reading a cube should not pull in scipy/tqdm.
        from luhdm import rate

        kw.setdefault("projection_kernel", self.projection_kernel)
        kw.setdefault("b_constrained_max", self.b_constrained_max)
        return rate.make_xsec(lamb, **kw)

    @property
    def f_dm_values(self):
        """DM-fraction hypotheses this file carries, ascending (e.g. [0.1, 1.0]).

        Pre-dual-f cubes carry only the baseline; ``[0.1]`` is then returned.
        """
        if self.layout == "axes":
            return [float(x) for x in self._f_dm_axis]
        v = self.attrs.get("f_dm_values")
        if v is None:
            return [float(self.attrs.get("f_x", F_DM_DEFAULT))]
        return [float(x) for x in np.atleast_1d(v)]

    @property
    def atmosphere_values(self):
        """Atmosphere hypotheses carried, as booleans (attenuation on/off)."""
        if self.layout == "axes":
            return [bool(x) for x in self._atmosphere_axis]
        return [True, False]

    # -- internal resolvers ----------------------------------------------- #
    def _resolve_group(self, group, atmosphere, default_group):
        """Normalise the (group, atmosphere) pair to a group name.

        ``group`` is the historical selector ('atm'/'noatm'/'halo') and
        ``atmosphere`` the axis-native one (True = attenuation applied). Passing
        neither uses this accessor's historical default; passing both is allowed
        only when they agree, so a caller can never silently get the other
        hypothesis than the one it named.
        """
        if group is not None and group not in _GROUP_H5:
            raise ValueError(
                f"unknown group {group!r}; expected 'atm', 'noatm' or 'halo'")
        if atmosphere is None:
            return default_group if group is None else group
        want = _ATM_GROUP[bool(atmosphere)]
        if group is not None and group != want:
            raise ValueError(
                f"group={group!r} and atmosphere={atmosphere!r} disagree; "
                f"atmosphere={bool(atmosphere)} means group={want!r}")
        return want

    def _f_dm_index(self, f_dm):
        """Index into ``axes/f_dm`` (axis layout), validated."""
        vals = self._f_dm_axis
        idx = np.where(np.isclose(vals, float(f_dm), rtol=1e-12, atol=0.0))[0]
        if not idx.size:
            raise ValueError(
                f"f_dm={f_dm!r} is not on this cube's f_dm axis "
                f"{[float(v) for v in vals]}")
        return int(idx[0])

    def _atmosphere_index(self, group):
        """Index into ``axes/atmosphere`` for a group name (axis layout)."""
        want = 1 if group == "atm" else 0
        idx = np.where(self._atmosphere_axis == want)[0]
        if not idx.size:
            raise ValueError(
                f"atmosphere={want} ({group}) is not on this cube's atmosphere "
                f"axis {self._atmosphere_axis.tolist()}")
        return int(idx[0])

    def _cube(self, quantity, group, f_dm):
        """(dataset, leading-index tuple) for one quantity+hypothesis.

        The single place the two layouts differ: in the axis layout the
        hypothesis is a pair of leading indices into one ``/results`` array; in
        the v3 layout it is the group name plus a dataset-name suffix. Callers
        index ``ds[prefix + rest]`` either way.
        """
        if quantity not in _QUANTITIES:
            raise ValueError(
                f"quantity must be one of {'/'.join(_QUANTITIES)}; "
                f"got {quantity!r}")
        if self.layout == "axes":
            return (self._file["results"][quantity],
                    (self._f_dm_index(f_dm), self._atmosphere_index(group)))
        if quantity == "n_transit":
            # never stored per f_DM in the v3 layout: it is exactly linear in
            # f_DM, so the caller scales the plane it reads back.
            self._check_f_dm_value(f_dm)
            return self._h5group(group)["n_transit"], ()
        return self._h5group(group)[quantity + self._f_dm_suffix(
            f_dm, group, quantity)], ()

    @staticmethod
    def _check_f_dm_value(f_dm):
        """Reject an f_DM that is not one of the hypotheses this loader knows."""
        if float(f_dm) not in _F_DM_SUFFIX:
            raise ValueError(
                f"f_dm={f_dm!r} is not a supported hypothesis; "
                f"known values: {sorted(_F_DM_SUFFIX)}")

    def _f_dm_suffix(self, f_dm, group, quantity):
        """Dataset-name suffix for ``f_dm`` in the v3 group layout.

        Returns "" for the baseline and "_f1" for the f_DM = 1 surface. Raises
        ValueError for an unsupported value, naming what the file does carry.
        """
        f = float(f_dm)
        avail = self.f_dm_values
        suffix = _F_DM_SUFFIX.get(f)
        if suffix is None:
            raise ValueError(
                f"f_dm={f_dm!r} is not a supported hypothesis; this cube "
                f"carries {avail}")
        if suffix and (quantity not in _PER_MODE
                       or quantity + suffix not in self._h5group(group)):
            raise ValueError(
                f"f_dm={f} is not available for {group}/{quantity} in this "
                f"cube (it carries {avail}); rebuild with the dual-f_DM "
                f"pipeline (scripts/build_release.py schema 2)")
        return suffix

    def _f_dm_scale(self, f_dm):
        """Linear flux scaling from the baseline to ``f_dm`` (n_transit etc.).

        Only used where a quantity is not stored per f_DM (the v3 layout's
        ``n_transit``, and the halo group in both layouts).
        """
        return float(f_dm) / F_DM_DEFAULT

    def _axis_for(self, group):
        if group in ("atm", None):
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
        name = _GROUP_H5[group]
        if name not in self._file:
            raise KeyError(
                f"this cube has no /{name} group (layout={self.layout!r}); "
                f"read it through the accessors, which map the atmosphere "
                f"hypothesis onto whichever layout the file uses")
        return self._file[name]

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
    def mass_plane(self, quantity, mode=None, lam="200um", group=None,
                   atmosphere=None, f_dm=F_DM_DEFAULT):
        """A (n_alpha, n_mass) plane at fixed lambda.

        ``quantity`` in ``{extremeness, mu, n_transit, status}`` for the
        atmosphere planes (``extremeness``/``mu``/``status`` need ``mode``;
        ``n_transit`` is mode-less), or ``{n_transit, bmax}`` for
        ``group='halo'`` (``mode`` must be ``None``).

        The hypothesis is named by ``f_dm`` (default 0.1) and ``atmosphere``
        (default True = attenuation applied); ``group='atm'``/``'noatm'`` is the
        historical spelling of the same atmosphere choice and still works.
        """
        group = self._resolve_group(group, atmosphere, "atm")
        if group == "halo":
            if mode is not None:
                raise ValueError("halo quantities are mode-less; pass mode=None")
            if quantity not in ("n_transit", "bmax"):
                raise ValueError(
                    f"halo quantity must be 'n_transit' or 'bmax'; got {quantity!r}")
            il = self.at_lambda(lam, "halo")
            plane = self._h5group("halo")[quantity][:, :, il]
            return plane * self._f_dm_scale(f_dm) if quantity == "n_transit" \
                else plane

        il = self.at_lambda(lam, group)
        ds, pre = self._cube(quantity, group, f_dm)
        if quantity == "n_transit":
            if mode is not None:
                raise ValueError("n_transit is mode-less; pass mode=None")
            plane = ds[pre + (slice(None), slice(None), il)]
            # stored per f_DM in the axis layout; scaled on read in the v3 one
            return plane if self.layout == "axes" \
                else plane * self._f_dm_scale(f_dm)
        return ds[pre + (self._mode_index(mode), slice(None), slice(None), il)]

    def lambda_plane(self, quantity, mode, mass=None, group=None,
                     atmosphere=None, f_dm=F_DM_DEFAULT):
        """A (n_alpha, n_finite) plane at fixed mass over the finite lambdas.

        ``mass=None`` uses :meth:`best_mass` for this mode/hypothesis. Not
        defined for the halo group; the trailing massless lambda column is
        excluded.
        """
        group = self._resolve_group(group, atmosphere, "atm")
        if group == "halo":
            raise ValueError("lambda_plane is defined for 'atm'/'noatm'; "
                             "got 'halo'")
        axset = self._axis_for(group)
        nfin = axset.n_finite
        im = self.best_mass(mode, group, f_dm=f_dm)[1] if mass is None \
            else self.at_mass(mass, group)
        ds, pre = self._cube(quantity, group, f_dm)
        if quantity == "n_transit":
            plane = ds[pre + (slice(None), im, slice(0, nfin))]
            return plane if self.layout == "axes" \
                else plane * self._f_dm_scale(f_dm)
        return ds[pre + (self._mode_index(mode), slice(None), im,
                         slice(0, nfin))]

    def composite(self, quantity="extremeness", lam="massless", group=None,
                  atmosphere=None, f_dm=F_DM_DEFAULT):
        """Per-mode maximum of a plane: ``np.maximum.reduce`` over the 3 modes.

        Historically defaults to the bare-halo (no-atmosphere) massless slice.
        """
        if quantity not in _PER_MODE:
            raise ValueError(
                f"composite needs a per-mode quantity {_PER_MODE}; got {quantity!r}")
        group = self._resolve_group(group, atmosphere, "noatm")
        il = self.at_lambda(lam, group)
        ds, pre = self._cube(quantity, group, f_dm)
        n_mode = self._mode.size
        planes = [ds[pre + (k, slice(None), slice(None), il)]
                  for k in range(n_mode)]
        return np.maximum.reduce(planes)

    # -- derived ---------------------------------------------------------- #
    def best_mass(self, mode, group=None, confidence=0.95, f_dm=F_DM_DEFAULT,
                  atmosphere=None):
        """(mass_gev, im): the mode's best mass via :func:`best_mass_index`.

        Memoised per ``(mode, group, confidence, f_dm)``. Loads the mode's
        finite-lambda extremeness cube once and delegates to the shared pure
        criterion.
        """
        group = self._resolve_group(group, atmosphere, "atm")
        if group == "halo":
            raise ValueError("best_mass is defined for 'atm'/'noatm'; "
                             "got 'halo'")
        key = (int(mode), group, float(confidence), float(f_dm))
        if key in self._best_mass_cache:
            return self._best_mass_cache[key]
        axset = self._axis_for(group)
        nfin = axset.n_finite
        ds, pre = self._cube("extremeness", group, f_dm)
        p_finite = ds[pre + (self._mode_index(mode), slice(None), slice(None),
                             slice(0, nfin))]
        im = best_mass_index(p_finite, axset.lambda_finite, axset.alpha_n,
                             confidence)
        result = (float(axset.mass_gev[im]), im)
        self._best_mass_cache[key] = result
        return result

    def excluded_alpha_band(self, mass, lam, mode=1, group=None,
                            confidence=0.95, f_dm=F_DM_DEFAULT,
                            atmosphere=None):
        """(lo, hi) coupling edges excluded at fixed (mass, lambda, mode).

        Delegates to :func:`luhdm.limits.excluded_band` on the 1-D
        ``p(alpha)`` column; ``(nan, nan)`` if nothing is excluded.
        """
        group = self._resolve_group(group, atmosphere, "atm")
        axset = self._axis_for(group)
        im = self.at_mass(mass, group)
        il = self.at_lambda(lam, group)
        ds, pre = self._cube("extremeness", group, f_dm)
        column = ds[pre + (self._mode_index(mode), slice(None), im, il)]
        # The level is pre-narrowed to the file's float32 storage precision so
        # a builder value exactly at the level survives the narrowing
        # (float32 -> float64 is exact, so this reproduces the float32 test
        # without touching optimum_interval).
        return limits.excluded_band(axset.alpha_n, column,
                                    level=float(np.float32(confidence)))

    def cell(self, mass, alpha, lam, mode=1, group=None, f_dm=F_DM_DEFAULT,
             atmosphere=None):
        """Full dump of one (mass, alpha, lambda, mode) cell as a dict.

        Includes the values at the resolved cell for the selected (f_dm,
        atmosphere) hypothesis, plus the halo values (``bmax``,
        ``n_transit_halo``) from the nearest halo cell at the same lambda. The
        hypothesis is echoed back in the dict.
        """
        group = self._resolve_group(group, atmosphere, "atm")
        axset = self._axis_for(group)
        ia = self.at_alpha(alpha, group)
        im = self.at_mass(mass, group)
        il = self.at_lambda(lam, group)
        imode = self._mode_index(mode)
        scale = self._f_dm_scale(f_dm)
        cells = {}
        for q in _QUANTITIES:
            ds, pre = self._cube(q, group, f_dm)
            if q == "n_transit":
                v = float(ds[pre + (ia, im, il)])
                cells[q] = v if self.layout == "axes" else v * scale
            else:
                cells[q] = ds[pre + (imode, ia, im, il)]
        # nearest halo cell at the same lambda index (halo shares the lambda axis)
        ia_h = self.at_alpha(alpha, "halo")
        im_h = self.at_mass(mass, "halo")
        halo = self._h5group("halo")
        return {
            "mass_gev": float(axset.mass_gev[im]),
            "alpha_n": float(axset.alpha_n[ia]),
            "lambda_m": float(axset.lambda_m[il]),
            "f_dm": float(f_dm),
            "atmosphere": group == "atm",
            "indices": (ia, im, il),
            "extremeness": float(cells["extremeness"]),
            "mu": float(cells["mu"]),
            "n_transit": float(cells["n_transit"]),
            "status": int(cells["status"]),
            "bmax": float(halo["bmax"][ia_h, im_h, il]),
            "n_transit_halo": float(halo["n_transit"][ia_h, im_h, il]) * scale,
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
