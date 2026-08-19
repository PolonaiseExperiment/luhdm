"""Standalone reader for the POLONAISE ultra-heavy dark-matter (UHDM) data release.

SPDX-License-Identifier: GPL-3.0-or-later

This reader is code, so it is GPL-3.0-or-later like the rest of the analysis
repository; the ``LICENSE`` file sitting next to it in the release directory is
CC BY 4.0 and covers the *data*, not this file. See section 13 of the release
``README.md``.

Copy this single file next to the release ``.h5`` and ``import luhdm_release``.
It needs **numpy** and **h5py** only (``pandas`` is imported lazily, inside
:meth:`Release.to_dataframe`, and is optional). It deliberately does *not*
import the ``luhdm`` analysis package, has no relative imports and no data
files, so it works from any directory on any machine that can open HDF5. If
your cluster will not let you install anything at all, read this file as
reference: every operation it performs is a few lines of plain numpy, and the
release ``README.md`` shows the equivalent bare-``h5py`` snippets.

Everything here is **schema-driven**: axis names, lengths, units, the mediator
tag table and the massless sentinel are read from the file's own attributes and
dimension scales. Nothing about the grid is hardcoded, so this reader keeps
working when the cube is regenerated with different axis lengths or values.

The cube
--------
``/results/<quantity>`` holds the analysis as a dense array over
``(f_dm, atmosphere, mode, alpha_n, mass_gev, lambda_m)``; ``n_transit`` has no
``mode`` axis because the expected number of transits does not depend on the
sensor mode. The two leading axes are hypothesis axes: ``f_dm`` is the fraction
of the local dark-matter density carried by this species (a pure flux
normalisation) and ``atmosphere`` is 1 when propagation through the
atmosphere/overburden is applied, 0 for the bare halo flux.

A release file may carry several values on those two axes or a single one. The
v9 release ships **one hypothesis per file** (``_A_f1_atm`` is ``f_dm = 1`` with
attenuation, ``_B_f0p1_noatm`` is ``f_dm = 0.1`` without), so both axes have
length 1 and the layout is otherwise unchanged. Note that ``f_dm_default``,
which every accessor falls back to when the caller names no fraction, is the
build-side baseline 0.1 and is *not* on file A's axis: pass ``f_dm=1.0``
explicitly when reading file A.

The mass window
---------------
Nothing in this reader applies the release's mass cut. The stored surfaces are
uncapped in impact parameter, so at very heavy masses they report exclusion
where the halo delivers essentially no transits through the apparatus, and the
release closes the region from the right with a separate, post-hoc flux cut
stored in the ``m_cut_*`` root attributes (``m_cut_10cm_f1_gev`` in file A,
``m_cut_10cm_f0.1_gev`` in file B), together with the ``N_req`` it assumes and
a one-paragraph derivation. :meth:`Release.excluded_band` returns every mass the
surfaces exclude; mask at ``mass_gev <= m_cut`` yourself. Section 5.4 of the
release ``README.md`` is the full statement.

The ``lambda_m`` axis is the finite mediator ranges in ascending order followed
by ``inf`` as the last element: that is the **massless** (Coulomb-like) slice,
for which ``axes/m_phi_gev`` is exactly ``0.0``. The number of finite entries is
the ``n_finite`` attribute of ``axes/lambda_m``; select the massless slice with
``lam='massless'`` (or ``lam=numpy.inf``).

Exclusion convention
--------------------
``extremeness`` is the probability that a background-free pseudo-experiment
under a hypothesis looks *less* extreme than the observed data. A point is
excluded at confidence ``C`` when ``extremeness >= C`` (95% CL -> 0.95). Along
a 1-D coupling scan the project quotes the *interpolated* crossing:
:func:`excluded_interval` reproduces it exactly (linear interpolation of the
level in ``log10(alpha_n)`` between the two bracketing grid points, saturating
at the ends of the scanned grid). :meth:`Release.excluded_band` applies it per
mass.

Status codes and NaN policy
---------------------------
``/results/status`` labels how each cell was obtained:

==== ===========================================================
code meaning
==== ===========================================================
0    Monte-Carlo extremeness computed (mu inside the MC window)
1    the cell raised an exception; ``extremeness`` and ``mu`` are NaN
2    ``mu`` below the MC floor -> shortcut, extremeness is exactly 0
3    ``mu`` above the MC cap -> shortcut, extremeness is exactly 1
     (overwhelmingly excluded)
4    the spectrum has no support, ``mu == 0``
==== ===========================================================

Codes 2, 3 and 4 are exact deterministic outcomes, not failures. Only code 1 is
a failure, and the project convention is that ``NaN >= C`` is False, i.e. a
status-1 cell reads as *not excluded*. This reader never changes that (the
published numbers must stay reproducible) but it never lets it pass silently
either: :meth:`Release.excluded_band` counts the undefined cells it saw,
returns the count, and warns unless you pass ``nan_policy='ignore'``.

Units of the detector lists
---------------------------
``detector/events_mode{n}`` is in **GeV** and ``detector/all_blips_mode{n}`` is
in **eV**: a factor 1e9 between two datasets in the same group.
:meth:`Release.events` and :meth:`Release.all_blips` return each list exactly as
stored, so divide the blip momenta by 1e9 before comparing or plotting them on
one axis. Every dataset carries its own ``units`` attribute.

Quickstart
----------
::

    import luhdm_release

    with luhdm_release.open_release("luhdm_datarelease_v10_A_f1_atm.h5") as rel:
        rel.summary()
        sl = rel.get("extremeness", mode=1, lam="200um")   # (alpha_n, mass_gev)
        band = rel.excluded_band(mode=1, lam="200um")
        print(band.mass_range, band.alpha_lo, band.alpha_hi)

From the shell, ``python luhdm_release.py luhdm_datarelease_v10_A_f1_atm.h5`` prints the
same summary.
"""

from __future__ import annotations

import json
import warnings

import h5py
import numpy as np

__all__ = [
    "STATUS_MEANINGS",
    "ExcludedBand",
    "Release",
    "Slice",
    "excluded_interval",
    "open_release",
]

__version__ = "1.0"

#: Human-readable meaning of each ``/results/status`` code. Mirrors the
#: ``description`` attribute stored on the dataset itself, which wins if the two
#: ever disagree (see :meth:`Release.status_legend`).
STATUS_MEANINGS = {
    0: "the optimum-interval Monte Carlo ran",
    1: "the cell raised; extremeness/mu/n_transit are NaN, and NaN reads as "
       "NOT excluded",
    2: "expected counts below the MC floor; extremeness is exactly 0",
    3: "expected counts above the MC cap; extremeness is exactly 1 (excluded)",
    4: "the spectrum has no support; extremeness is exactly 0",
}

# Roles the user may select by, mapped onto whatever the file calls that axis.
# The match is by substring on the dimension label, so the halo maps (whose axes
# are 'alpha_halo_n' / 'mass_halo_gev') and any future renaming still resolve.
_ROLE_PATTERNS = {
    "f_dm": ("f_dm",),
    "atmosphere": ("atmosphere",),
    "mode": ("mode",),
    "alpha": ("alpha",),
    "mass": ("mass",),
    "lam": ("lambda",),
}

# How a physical value is matched onto each role's axis.
#   "exact"       -- np.isclose with a tight rtol; a miss is an error
#   "nearest_log" -- nearest grid point in log10 (the analysis convention for
#                    the continuous scan axes, which are log-spaced)
_ROLE_MATCH = {
    "f_dm": "exact",
    "atmosphere": "exact",
    "mode": "exact",
    "alpha": "nearest_log",
    "mass": "nearest_log",
    "lam": "exact",
}


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


def _fmt(x, n=4):
    """Compact repr of a float for error messages and summaries."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if not np.isfinite(x):
        return str(x)
    if x == 0.0:
        return "0"
    return f"{x:.{n}g}"


def _fmt_list(values, limit=12):
    """``[a, b, ... , z]`` with an ellipsis when the axis is long."""
    vals = [_fmt(v) for v in np.asarray(values).ravel()]
    if len(vals) <= limit:
        return "[" + ", ".join(vals) + "]"
    head = ", ".join(vals[: limit // 2])
    tail = ", ".join(vals[-(limit // 2):])
    return f"[{head}, ... ({len(vals)} values) ..., {tail}]"


# --------------------------------------------------------------------------- #
# The exclusion convention, as a pure function
# --------------------------------------------------------------------------- #
def excluded_interval(params, extremeness, level=0.95):
    """``(low, high)`` edges of the excluded interval along a 1-D scan.

    This is the project's exclusion convention written out in plain numpy, and
    is what :meth:`Release.excluded_band` calls. ``params`` is the scanned
    parameter (positive and increasing -- for this release the coupling
    ``alpha_n``) and ``extremeness`` the matching extremeness values.

    The excluded set is the level set ``extremeness >= level``. Its edges are
    *log-interpolated*: the level crossing is found by linear interpolation in
    ``log10(params)`` between the two bracketing grid points, so the answer does
    not jump by a whole grid step. When the excluded set already includes the
    first (last) grid point there is nothing to bracket and the edge saturates
    at that end of the scanned grid -- such an edge is a property of the grid,
    not a measurement.

    ``NaN`` extremeness (status 1) compares False and therefore reads as *not
    excluded*; this is deliberate and matches the published analysis. Use
    :meth:`Release.excluded_band`, which counts and reports those cells, rather
    than calling this function blind on a column you have not inspected.

    Returns ``(nan, nan)`` when nothing reaches ``level``.
    """
    params = np.asarray(params, dtype=float)
    ps = np.asarray(extremeness, dtype=float)
    # The file stores extremeness as float32, so the level set is taken at the
    # storage precision: a builder value exactly at the level narrows to just
    # below it in float32 and would otherwise be dropped from the excluded set
    # (float32 -> float64 is exact, so this reproduces the float32 comparison).
    level = float(np.float32(level))
    above = ps >= level
    if not above.any():
        return np.nan, np.nan
    idx = np.where(above)[0]
    if idx[0] > 0:
        lo = np.interp(level, ps[idx[0] - 1: idx[0] + 1],
                       np.log10(params[idx[0] - 1: idx[0] + 1]))
    else:
        lo = np.log10(params[0])
    if idx[-1] < len(params) - 1:
        hi = np.interp(-level, -ps[idx[-1]: idx[-1] + 2],
                       np.log10(params[idx[-1]: idx[-1] + 2]))
    else:
        hi = np.log10(params[-1])
    return 10 ** lo, 10 ** hi


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
class Slice:
    """A numpy array read out of the cube, carrying the axes it is indexed by.

    Attributes
    ----------
    values : ndarray
        The data. Axes that were pinned to a single physical value are dropped,
        so a 6-D cube with ``mode`` and ``lam`` pinned comes back 4-D.
    dims : tuple of str
        Names of the surviving axes, in the array's own order.
    axes : dict
        ``{dim_name: 1-D ndarray}`` -- the coordinate values of each surviving
        axis, straight from ``/axes/<dim_name>``.
    selection : dict
        ``{dim_name: value}`` for every axis that was pinned, giving the value
        actually used (i.e. after nearest-grid-point resolution).
    indices : dict
        ``{dim_name: int}`` grid index for every pinned axis.
    units, description, name : str
        Copied from the dataset's attributes.
    """

    def __init__(self, values, dims, axes, selection, indices,
                 units="", description="", name=""):
        self.values = values
        self.dims = tuple(dims)
        self.axes = axes
        self.selection = selection
        self.indices = indices
        self.units = units
        self.description = description
        self.name = name

    def __array__(self, dtype=None, copy=None):
        arr = self.values if dtype is None else self.values.astype(dtype)
        return np.array(arr, copy=True) if copy else arr

    @property
    def shape(self):
        return self.values.shape

    def axis(self, dim):
        """Coordinate array of one surviving axis."""
        if dim not in self.axes:
            raise KeyError(
                f"{dim!r} is not a surviving axis of this slice; "
                f"it has {list(self.axes)}")
        return self.axes[dim]

    def __repr__(self):
        sel = ", ".join(f"{k}={_fmt(v)}" for k, v in self.selection.items())
        dims = ", ".join(f"{d}={len(self.axes[d])}" for d in self.dims)
        unit = f" [{self.units}]" if self.units else ""
        return (f"<Slice {self.name}{unit} ({dims}) "
                f"at {sel or 'full cube'}>")


class ExcludedBand:
    """Result of :meth:`Release.excluded_band`.

    Attributes
    ----------
    mass_gev : ndarray
        The mass axis the band was evaluated on.
    alpha_lo, alpha_hi : ndarray
        Lower and upper excluded coupling edge per mass, ``NaN`` where nothing
        is excluded at that mass. See :func:`excluded_interval` for the
        interpolation convention.
    mass_range : tuple
        ``(mass_min, mass_max)`` over the masses with a non-empty band, or
        ``(nan, nan)``.
    saturated_lo, saturated_hi : ndarray of bool
        True where the corresponding edge sits exactly on the first/last
        scanned coupling, i.e. the exclusion runs off the end of the grid and
        the edge is a grid artefact rather than a measured boundary.
    n_undefined : ndarray of int
        Per mass, how many coupling grid points had ``NaN`` extremeness
        (status 1). Those cells count as *not excluded*; a non-zero entry means
        the band at that mass was computed from an incomplete column.
    level : float
        The confidence level used.
    selection : dict
        The hypothesis the band belongs to (f_dm, atmosphere, mode, lambda).
    """

    def __init__(self, mass_gev, alpha_lo, alpha_hi, saturated_lo,
                 saturated_hi, n_undefined, level, selection):
        self.mass_gev = mass_gev
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.saturated_lo = saturated_lo
        self.saturated_hi = saturated_hi
        self.n_undefined = n_undefined
        self.level = level
        self.selection = selection

    @property
    def any_excluded(self):
        return bool(np.isfinite(self.alpha_lo).any())

    @property
    def mass_range(self):
        ok = np.isfinite(self.alpha_lo)
        if not ok.any():
            return (np.nan, np.nan)
        m = self.mass_gev[ok]
        return (float(m.min()), float(m.max()))

    def __iter__(self):
        """Unpack as ``alpha_lo, alpha_hi = band``."""
        yield self.alpha_lo
        yield self.alpha_hi

    def __repr__(self):
        lo, hi = self.mass_range
        sel = ", ".join(f"{k}={_fmt(v)}" for k, v in self.selection.items())
        return (f"<ExcludedBand {int(self.level * 100)}% CL at {sel}: "
                f"{int(np.isfinite(self.alpha_lo).sum())}/"
                f"{self.mass_gev.size} masses excluded, "
                f"mass range {_fmt(lo)}..{_fmt(hi)} GeV>")


# --------------------------------------------------------------------------- #
# Open
# --------------------------------------------------------------------------- #
def open_release(path):
    """Open a release HDF5 read-only and return a :class:`Release`.

    ``path`` is required: this reader has no built-in location, so it works
    wherever you copied the file. Use it as a context manager, or call
    :meth:`Release.close` when you are done::

        with open_release("luhdm_datarelease_v10_A_f1_atm.h5") as rel:
            ...
    """
    f = h5py.File(str(path), "r")
    try:
        fmt = _py(f.attrs.get("file_format"))
        if fmt != "luhdm-datarelease":
            raise ValueError(
                f"{path}: file_format attribute is {fmt!r}, expected "
                f"'luhdm-datarelease'. This does not look like a POLONAISE "
                f"UHDM data release.")
        if "results" not in f or "axes" not in f:
            raise ValueError(
                f"{path}: expected top-level 'axes' and 'results' groups; "
                f"found {list(f)}. Older cubes used an /atm + /noatm layout "
                f"and are not readable with this reader.")
    except Exception:
        f.close()
        raise
    return Release(f, path)


class Release:
    """Read-only handle on one data-release HDF5 file.

    Construct it with :func:`open_release`. Axes are read eagerly (they are
    tiny); the cubes are read lazily as HDF5 hyperslabs, so pinning axes keeps
    memory small.
    """

    def __init__(self, f, path=""):
        self._f = f
        self.path = str(path)
        #: Root attributes as plain Python objects (version tag, exposure,
        #: impact-parameter cap, git commit, input hashes, ...).
        self.attrs = {k: _py(v) for k, v in f.attrs.items()}
        self._axes = {name: f["axes"][name][:] for name in f["axes"]}
        self._axis_attrs = {name: {k: _py(v) for k, v in f["axes"][name].attrs.items()}
                            for name in f["axes"]}

    # -- lifecycle -------------------------------------------------------- #
    def close(self):
        """Close the underlying HDF5 file (idempotent)."""
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return f"<Release {self.path!r} ({self.version_tag})>"

    # -- metadata --------------------------------------------------------- #
    @property
    def version_tag(self):
        """Human-readable cube version, e.g. ``'v9.1-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mc2tier'``."""
        return str(self.attrs.get("version_tag")
                   or f"v{self.attrs.get('version', '?')}")

    @property
    def exposure_s(self):
        """Total live-time exposure in seconds."""
        if "detector" in self._f and "exposure_s" in self._f["detector"]:
            return float(self._f["detector"]["exposure_s"][()])
        return float(self.attrs["t_exposure_s"])

    @property
    def b_constrained_max_m(self):
        """Impact-parameter cap in metres, or ``None`` if the cube is uncapped.

        When set, the outer limit of the impact-parameter integral is
        ``min(b_constrained_max, b_max(q))`` in both the differential cross
        section and the geometric transit reach: flybys further away than this
        are not counted.
        """
        v = self.attrs.get("b_constrained_max_m")
        return None if v is None or not np.isfinite(v) else float(v)

    @property
    def confidence_recommended(self):
        """Confidence level the release is quoted at (0.95 unless overridden)."""
        return float(self.attrs.get("confidence_recommended", 0.95))

    @property
    def quantities(self):
        """Names of the datasets under ``/results``."""
        return sorted(self._f["results"])

    @property
    def modes(self):
        """Sensor mode labels carried by the cube."""
        return [int(v) for v in self._axes["mode"]]

    @property
    def f_dm_values(self):
        """Dark-matter fraction hypotheses carried, in axis order."""
        return [float(v) for v in self._axes["f_dm"]]

    @property
    def atmosphere_values(self):
        """Atmosphere hypotheses carried, as booleans, in axis order."""
        return [bool(v) for v in self._axes["atmosphere"]]

    @property
    def lambda_m(self):
        """Full mediator-range axis in metres (finite ascending, then ``inf``)."""
        return self._axes["lambda_m"]

    @property
    def n_finite_lambda(self):
        """Number of finite entries on the lambda axis (the rest is massless)."""
        a = self._axis_attrs.get("lambda_m", {})
        if "n_finite" in a:
            return int(a["n_finite"])
        return int(np.count_nonzero(np.isfinite(self._axes["lambda_m"])))

    @property
    def lambda_finite(self):
        """The finite part of the lambda axis (massless sentinel dropped)."""
        return self._axes["lambda_m"][: self.n_finite_lambda]

    @property
    def lambda_tags(self):
        """``{tag: lambda_m}`` named mediator ranges, read from the file.

        Each tag value is an exact float, so a tag that is on the axis resolves
        to a pure integer slice. The table is inherited from the parent scan and
        may name ranges this file does not carry -- in the v9 release only three
        of its eight tags are on the axis -- so a tag is not a promise that the
        slice is here; filter against :meth:`axis` (``'lambda_m'``) if you are
        iterating. Asking for an absent one raises and lists the axis.
        ``'massless'`` is added here for the ``inf`` sentinel, which the file
        stores as a value rather than a tag.
        """
        tags = {}
        raw = self._axis_attrs.get("lambda_m", {}).get("tags_json")
        if raw:
            tags.update({str(k): float(v) for k, v in json.loads(raw).items()})
        tags["massless"] = np.inf
        return tags

    def axis(self, name):
        """Coordinate array of one axis, by its name under ``/axes``."""
        if name not in self._axes:
            raise KeyError(
                f"no axis {name!r} in this file; it has {sorted(self._axes)}")
        return self._axes[name]

    def axis_names(self):
        """All axis names present under ``/axes``."""
        return sorted(self._axes)

    def status_legend(self):
        """``{code: meaning}``, preferring the description stored in the file.

        The file's own ``/results/status`` ``description`` attribute is the
        authority; :data:`STATUS_MEANINGS` is the fallback / long form.
        """
        legend = dict(STATUS_MEANINGS)
        desc = _py(self._f["results"]["status"].attrs.get("description", ""))
        for token in str(desc).replace(",", " ").split():
            if "=" in token:
                code, _, text = token.partition("=")
                try:
                    code = int(code)
                except ValueError:
                    continue
                long = legend.get(code, "")
                legend[code] = f"{text}: {long}" if long else text
        return legend

    def provenance(self):
        """Provenance-ish root attributes as a dict (hashes, git, packages).

        Everything needed to say where the numbers came from: the git commit and
        dirty flag of the analysis code, the package versions, the SHA256 of
        every input file, the Monte-Carlo fidelity settings and the RNG seed.
        """
        keys = [k for k in self.attrs
                if k.startswith(("git_", "events_", "efficiency_", "pkg_", "fid_"))
                or k in ("created", "seed", "version", "version_tag",
                         "schema_version", "file_format", "layout",
                         "packages_json", "inputs_json", "events_dir")]
        return {k: self.attrs[k] for k in sorted(keys)}

    # -- detector inputs -------------------------------------------------- #
    def events(self, mode):
        """Observed impulse candidates for one sensor mode, in **GeV**.

        This is the analysis event list the limit is set on -- the momentum
        kicks surviving the full selection. Note the unit difference against
        :meth:`all_blips`, which is in eV.
        """
        return self._f["detector"][f"events_mode{int(mode)}"][:]

    def all_blips(self, mode):
        """Pre-selection impulse momenta for one mode, in **eV**.

        Every reconstructed transient above the analysis threshold, before the
        quality selection that produces :meth:`events`. Returned exactly as
        stored, so **divide by 1e9 to compare with** :meth:`events`, which is in
        GeV. Context only; the limit is not set on this list.
        """
        return self._f["detector"][f"all_blips_mode{int(mode)}"][:]

    def efficiency_curve(self, mode, df=None):
        """``(q_gev, efficiency)`` measured detection efficiency for one mode.

        ``df`` selects the degrees-of-freedom hypothesis of the efficiency fit;
        it defaults to the one the analysis used (root attribute ``df``).
        """
        if df is None:
            df = int(self.attrs.get("df", 3))
        det = self._f["detector"]
        return (det[f"q_gev_{int(mode)}"][:],
                det[f"eff_{int(mode)}_df{int(df)}"][:])

    def raw(self, path):
        """Escape hatch: any dataset by its HDF5 path, as a numpy array.

        e.g. ``rel.raw('reference_curves/drdq_200um')`` or
        ``rel.raw('halo/bmax')``.
        """
        obj = self._f[path]
        if not isinstance(obj, h5py.Dataset):
            raise KeyError(f"{path!r} is a group, not a dataset; "
                           f"it contains {list(obj)}")
        return obj[()]

    # -- index resolution -------------------------------------------------- #
    def _dataset(self, quantity):
        """Resolve a quantity name (or full HDF5 path) to a dataset."""
        if quantity in self._f["results"]:
            return self._f["results"][quantity]
        if "/" in quantity and quantity in self._f:
            obj = self._f[quantity]
            if isinstance(obj, h5py.Dataset):
                return obj
        raise KeyError(
            f"unknown quantity {quantity!r}; /results holds {self.quantities}. "
            f"You can also pass a full dataset path such as 'halo/bmax'.")

    @staticmethod
    def _dims_of(ds):
        """Axis names of a dataset, from its dimension labels / ``axes`` attr."""
        labels = ds.attrs.get("DIMENSION_LABELS")
        if labels is not None:
            return [_py(x) for x in labels]
        axes = _py(ds.attrs.get("axes", ""))
        if axes:
            return [s.strip() for s in axes.split(",")]
        raise ValueError(
            f"dataset {ds.name} carries no DIMENSION_LABELS or 'axes' "
            f"attribute, so its axes cannot be identified")

    def _dim_for_role(self, role, dims):
        """Which of ``dims`` plays ``role`` ('mass', 'alpha', 'lam', ...)."""
        pats = _ROLE_PATTERNS[role]
        hits = [d for d in dims if any(p in d for p in pats)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            return None
        raise ValueError(
            f"ambiguous selector {role!r}: it matches axes {hits}; "
            f"select by the exact axis name instead")

    def _index_exact(self, name, value):
        """Exact index on an axis, with an error that lists what is available."""
        axis = self._axes[name]
        if name == "atmosphere":
            target = 1 if bool(value) else 0
            idx = np.where(np.asarray(axis, dtype=int) == target)[0]
            if idx.size:
                return int(idx[0])
            raise KeyError(
                f"atmosphere={bool(value)} (axis value {target}) is not in this "
                f"cube; axes/atmosphere = {axis.tolist()} "
                f"(1 = attenuation applied, 0 = bare halo)")
        if np.issubdtype(axis.dtype, np.integer):
            idx = np.where(axis == int(value))[0]
            if idx.size:
                return int(idx[0])
            raise KeyError(
                f"{name}={value!r} is not on the axis; available: "
                f"{axis.tolist()}")
        target = float(value) if not isinstance(value, str) else value
        if isinstance(target, str):
            raise KeyError(f"{name}={value!r}: expected a number")
        if np.isinf(target):
            idx = np.where(np.isinf(axis) & (np.sign(axis) == np.sign(target)))[0]
        else:
            rtol = 1e-9 if name == "lambda_m" else 1e-12
            idx = np.where(np.isclose(axis, target, rtol=rtol, atol=0.0))[0]
        if idx.size:
            return int(idx[0])
        if name == "lambda_m":
            loga = np.log10(target) if target > 0 else -np.inf
            axis_log = np.where(np.isfinite(axis) & (axis > 0),
                                np.log10(axis), np.inf)
            order = np.argsort(np.abs(axis_log - loga))[:3]
            near = [float(x) for x in axis[order]]
            raise KeyError(
                f"lambda {target!r} m is not an exact value on axes/lambda_m "
                f"(rtol 1e-9). Nearest axis values (m): {near}. "
                f"Named tags: {sorted(self.lambda_tags)}. "
                f"The axis has {axis.size} entries spanning "
                f"{_fmt(self.lambda_finite[0])}..{_fmt(self.lambda_finite[-1])} m "
                f"plus the massless sentinel inf.")
        raise KeyError(
            f"{name}={value!r} is not on the axis; available: "
            f"{_fmt_list(axis)}")

    def _index_nearest_log(self, name, value):
        """Nearest grid point in log10 -- the analysis convention for log axes."""
        axis = np.asarray(self._axes[name], dtype=float)
        v = float(value)
        if v <= 0 or not np.isfinite(v):
            raise ValueError(
                f"{name}={value!r} must be a positive finite number "
                f"(the axis is log-spaced over "
                f"{_fmt(axis.min())}..{_fmt(axis.max())})")
        if v < axis.min() or v > axis.max():
            warnings.warn(
                f"{name}={_fmt(v)} is outside the scanned range "
                f"{_fmt(axis.min())}..{_fmt(axis.max())}; clamping to the "
                f"nearest grid point {_fmt(axis[0] if v < axis.min() else axis[-1])}",
                stacklevel=5)
        return int(np.argmin(np.abs(np.log10(axis) - np.log10(v))))

    def _index(self, name, role, value):
        if _ROLE_MATCH[role] == "nearest_log":
            return self._index_nearest_log(name, value)
        return self._index_exact(name, value)

    def resolve(self, quantity, **selection):
        """``{axis_name: index}`` for a selection, without reading any data.

        Useful for checking what ``mass=1e8`` actually snapped to, or for
        indexing the raw h5py dataset yourself.
        """
        ds = self._dataset(quantity)
        dims = self._dims_of(ds)
        out = {}
        for role, value in selection.items():
            if role not in _ROLE_PATTERNS:
                raise TypeError(
                    f"unknown selector {role!r}; valid selectors are "
                    f"{sorted(_ROLE_PATTERNS)}")
            if value is None:
                continue
            name = self._dim_for_role(role, dims)
            if name is None:
                raise ValueError(
                    f"{quantity!r} has no {role!r} axis (its axes are {dims}); "
                    f"drop {role}= from the selection. "
                    + ("n_transit is mode-independent by construction."
                       if role == "mode" else ""))
            if role == "lam" and isinstance(value, str):
                tags = self.lambda_tags
                if value not in tags:
                    raise KeyError(
                        f"unknown lambda tag {value!r}; this file names "
                        f"{sorted(tags)}. You can also pass a value in metres.")
                value = tags[value]
            out[name] = self._index(name, role, value)
        return out

    # -- reads ------------------------------------------------------------- #
    def get(self, quantity, f_dm=None, atmosphere=None, mode=None,
            alpha=None, mass=None, lam=None):
        """Read a slice of a cube, selecting axes by physical value.

        Parameters
        ----------
        quantity : str
            ``'extremeness'``, ``'mu'``, ``'status'``, ``'n_transit'`` (anything
            under ``/results``), or a full dataset path like ``'halo/bmax'``.
        f_dm : float, optional
            Dark-matter fraction hypothesis, e.g. ``0.1`` or ``1.0``. Must be an
            exact axis value.
        atmosphere : bool, optional
            ``True`` = attenuation through the atmosphere/overburden applied,
            ``False`` = bare halo flux.
        mode : int, optional
            Sensor mode (1, 2, 3). Not accepted for ``n_transit``, which has no
            mode axis.
        alpha, mass : float, optional
            Coupling ``alpha_n`` and dark-matter mass in GeV. These are
            *continuous* log-spaced scan axes, so the nearest grid point in
            log10 is used; :attr:`Slice.selection` reports the value you got.
        lam : float or str, optional
            Mediator range in metres (exact axis value required), a named tag
            such as ``'200um'``, or ``'massless'`` / ``numpy.inf`` for the
            massless slice.

        Any parameter left as ``None`` keeps that whole axis.

        Returns
        -------
        Slice
            ``.values`` is the numpy array with the pinned axes dropped;
            ``.dims`` and ``.axes`` describe what is left.

        Notes
        -----
        ``extremeness`` and ``mu`` are ``NaN`` wherever ``status == 1``; read
        ``status`` alongside if that matters to you.
        """
        ds = self._dataset(quantity)
        dims = self._dims_of(ds)
        sel = dict(f_dm=f_dm, atmosphere=atmosphere, mode=mode,
                   alpha=alpha, mass=mass, lam=lam)
        indices = self.resolve(quantity, **sel)
        key = tuple(indices.get(d, slice(None)) for d in dims)
        values = ds[key]
        keep = [d for d in dims if d not in indices]
        return Slice(
            values=values,
            dims=keep,
            axes={d: self._axes[d] for d in keep if d in self._axes},
            selection={d: (bool(self._axes[d][i]) if d == "atmosphere"
                           else self._axes[d][i].item())
                       for d, i in indices.items()},
            indices=indices,
            units=_py(ds.attrs.get("units", "")),
            description=_py(ds.attrs.get("description", "")),
            name=quantity,
        )

    def extremeness(self, **selection):
        """:meth:`get` for ``extremeness`` (optimum-interval confidence)."""
        return self.get("extremeness", **selection)

    def mu(self, **selection):
        """:meth:`get` for ``mu`` (expected detected signal counts)."""
        return self.get("mu", **selection)

    def status(self, **selection):
        """:meth:`get` for ``status`` (see :data:`STATUS_MEANINGS`)."""
        return self.get("status", **selection)

    def n_transit(self, **selection):
        """:meth:`get` for ``n_transit`` (expected in-reach flybys, mode-less)."""
        return self.get("n_transit", **selection)

    def cell(self, mode, alpha, mass, lam, f_dm=None, atmosphere=True):
        """Every quantity at one fully pinned cell, as a plain dict.

        Handy for spot checks: it echoes back the resolved grid values, so you
        can see exactly which cell ``mass=1e8`` landed on.
        """
        f_dm = self._default_f_dm(f_dm)
        base = dict(f_dm=f_dm, atmosphere=atmosphere, alpha=alpha,
                    mass=mass, lam=lam)
        out = {}
        for q in self.quantities:
            has_mode = self._dim_for_role("mode", self._dims_of(self._dataset(q)))
            sl = self.get(q, mode=mode if has_mode else None, **base)
            out[q] = sl.values.item()
            out.setdefault("_resolved", sl.selection)
        resolved = out.pop("_resolved")
        out.update({k: v for k, v in resolved.items()})
        out["mode"] = int(mode)
        out["status_meaning"] = self.status_legend().get(int(out["status"]), "?")
        return out

    def _default_f_dm(self, f_dm):
        if f_dm is not None:
            return f_dm
        return float(self.attrs.get("f_dm_default", self.f_dm_values[0]))

    # -- the exclusion boundary ------------------------------------------- #
    def excluded_band(self, mode, lam, f_dm=None, atmosphere=True,
                      confidence=None, nan_policy="warn"):
        """Excluded coupling band per mass, at the project's 95% CL convention.

        Scans ``alpha_n`` at every mass for the chosen hypothesis and takes the
        ``extremeness >= confidence`` level set, log-interpolating each edge
        between the two bracketing grid points (see :func:`excluded_interval`).

        The band covers every mass on the grid. It does **not** stop at the
        release's flux cut ``m_cut`` (see the module docstring): mask the result
        at ``mass_gev <= m_cut`` before quoting or plotting it.

        Parameters
        ----------
        mode : int
            Sensor mode.
        lam : float or str
            Mediator range in metres, a named tag, or ``'massless'``.
        f_dm : float, optional
            Dark-matter fraction; defaults to the file's ``f_dm_default``.
        atmosphere : bool
            ``True`` (default) = attenuation applied.
        confidence : float, optional
            Confidence level; defaults to the file's ``confidence_recommended``
            (0.95).
        nan_policy : {'warn', 'raise', 'ignore'}
            What to do about cells with ``NaN`` extremeness (``status == 1``).
            Those cells compare False and so read as *not excluded*, which is
            the published convention and is never changed here. ``'warn'``
            (default) emits a :class:`RuntimeWarning` naming how many were seen,
            ``'raise'`` refuses to return a band computed from an incomplete
            column, ``'ignore'`` is silent. The count is always available as
            :attr:`ExcludedBand.n_undefined`.

        Returns
        -------
        ExcludedBand
            ``band.alpha_lo`` / ``band.alpha_hi`` are per-mass arrays and unpack
            as ``lo, hi = band``; ``band.mass_range`` is the excluded mass span.
        """
        if nan_policy not in ("warn", "raise", "ignore"):
            raise ValueError(
                f"nan_policy must be 'warn', 'raise' or 'ignore'; "
                f"got {nan_policy!r}")
        level = self.confidence_recommended if confidence is None \
            else float(confidence)
        f_dm = self._default_f_dm(f_dm)
        sl = self.get("extremeness", f_dm=f_dm, atmosphere=atmosphere,
                      mode=mode, lam=lam)
        # surviving axes are (alpha_n, mass_gev) in the cube's own order
        alpha_dim = self._dim_for_role("alpha", sl.dims)
        mass_dim = self._dim_for_role("mass", sl.dims)
        p = np.asarray(sl.values, dtype=float)
        if sl.dims.index(alpha_dim) != 0:
            p = np.moveaxis(p, sl.dims.index(alpha_dim), 0)
        alpha = np.asarray(sl.axes[alpha_dim], dtype=float)
        mass = np.asarray(sl.axes[mass_dim], dtype=float)

        n_mass = mass.size
        lo = np.full(n_mass, np.nan)
        hi = np.full(n_mass, np.nan)
        for im in range(n_mass):
            lo[im], hi[im] = excluded_interval(alpha, p[:, im], level=level)
        n_undef = np.isnan(p).sum(axis=0).astype(int)

        eps = 1e-12
        sat_lo = np.isfinite(lo) & (np.abs(lo / alpha[0] - 1.0) < eps)
        sat_hi = np.isfinite(hi) & (np.abs(hi / alpha[-1] - 1.0) < eps)

        total_undef = int(n_undef.sum())
        if total_undef and nan_policy != "ignore":
            msg = (f"{total_undef} of {p.size} extremeness cells are NaN "
                   f"(status==1, the cell raised) in this "
                   f"(mode={mode}, lam={lam!r}, f_dm={f_dm}, "
                   f"atmosphere={bool(atmosphere)}) plane. They compare False "
                   f"against the level and therefore read as NOT excluded, "
                   f"which is the published convention; the affected masses are "
                   f"in ExcludedBand.n_undefined.")
            if nan_policy == "raise":
                raise ValueError(msg + " Pass nan_policy='warn' or 'ignore' "
                                       "to proceed anyway.")
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return ExcludedBand(
            mass_gev=mass, alpha_lo=lo, alpha_hi=hi,
            saturated_lo=sat_lo, saturated_hi=sat_hi, n_undefined=n_undef,
            level=level,
            selection={"f_dm": f_dm, "atmosphere": bool(atmosphere),
                       "mode": int(mode),
                       "lambda_m": sl.selection.get("lambda_m")},
        )

    # -- tidy table -------------------------------------------------------- #
    def to_dataframe(self, f_dm=None, atmosphere=True, mode=None, lam=None,
                     mass=None, alpha=None, confidence=None, max_rows=2_000_000):
        """Tidy long-format ``pandas.DataFrame`` for a chosen hypothesis.

        One row per surviving grid cell, with the physical coordinates as
        columns alongside every quantity. ``pandas`` is imported here and only
        here, so the rest of this module works without it.

        The selectors are the same as :meth:`get`; anything left ``None`` is
        expanded into rows. ``f_dm`` defaults to the file's ``f_dm_default`` and
        ``atmosphere`` to ``True`` so a bare call does not silently mix the four
        hypotheses.

        Columns: ``f_dm, atmosphere, mode, alpha_n, mass_gev, lambda_m,
        m_phi_gev, extremeness, mu, status, status_meaning, n_transit,
        excluded``. ``excluded`` is ``extremeness >= confidence`` and is
        ``False`` where the extremeness is NaN -- the ``status`` column is how
        you tell those apart.

        Raises ``MemoryError`` if the selection would exceed ``max_rows``; pin
        another axis (or raise the limit deliberately).
        """
        try:
            import pandas as pd  # noqa: PLC0415  (optional dep, lazy on purpose)
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() is the only method that needs pandas "
                "('pip install pandas'). Everything else in this reader runs "
                "on numpy and h5py alone; get() returns the same numbers as a "
                "numpy array with its axes attached."
            ) from exc

        level = self.confidence_recommended if confidence is None \
            else float(confidence)
        f_dm = self._default_f_dm(f_dm)
        sel = dict(f_dm=f_dm, atmosphere=atmosphere, mode=mode,
                   alpha=alpha, mass=mass, lam=lam)

        ext = self.get("extremeness", **sel)
        n_rows = int(np.prod(ext.values.shape)) if ext.values.shape else 1
        if n_rows > max_rows:
            raise MemoryError(
                f"this selection would produce {n_rows:,} rows (limit "
                f"{max_rows:,}). Pin another axis (mode=, lam=, mass=) or pass "
                f"a larger max_rows explicitly.")
        mu = self.get("mu", **sel)
        status = self.get("status", **sel)
        nt_sel = {k: v for k, v in sel.items() if k != "mode"}
        nt = self.get("n_transit", **nt_sel)

        dims = list(ext.dims)
        grids = np.meshgrid(*[ext.axes[d] for d in dims], indexing="ij")
        data = {}
        for d, g in zip(dims, grids):
            data[d] = g.ravel()
        # pinned coordinates become constant columns
        for d, v in ext.selection.items():
            data[d] = np.full(n_rows, v)
        if "atmosphere" in data:
            data["atmosphere"] = np.asarray(data["atmosphere"], dtype=bool)

        # the mediator mass shares the lambda axis; carry it along for convenience
        lam_axis = self._axes["lambda_m"]
        mphi_axis = self._axes.get("m_phi_gev")
        if mphi_axis is not None:
            if "lambda_m" in dims:
                lut = dict(zip(lam_axis.tolist(), mphi_axis.tolist()))
                data["m_phi_gev"] = np.array([lut[v] for v in data["lambda_m"]])
            else:
                i = ext.indices["lambda_m"]
                data["m_phi_gev"] = np.full(n_rows, float(mphi_axis[i]))

        data["extremeness"] = np.asarray(ext.values, dtype=float).ravel()
        data["mu"] = np.asarray(mu.values, dtype=float).ravel()
        st = np.asarray(status.values, dtype=np.uint8).ravel()
        data["status"] = st
        legend = self.status_legend()
        data["status_meaning"] = np.array([legend.get(int(s), "?") for s in st])
        # n_transit has no mode axis: broadcast it back over the mode axis
        nt_vals = np.asarray(nt.values, dtype=float)
        if "mode" in dims:
            axis = dims.index("mode")
            nt_vals = np.repeat(np.expand_dims(nt_vals, axis),
                                len(ext.axes["mode"]), axis=axis)
        data["n_transit"] = nt_vals.ravel()
        # level set at storage precision (see excluded_interval)
        data["excluded"] = data["extremeness"] >= np.float32(level)

        order = [c for c in ("f_dm", "atmosphere", "mode", "alpha_n", "mass_gev",
                             "lambda_m", "m_phi_gev", "extremeness", "mu",
                             "status", "status_meaning", "n_transit", "excluded")
                 if c in data]
        order += [c for c in data if c not in order]
        return pd.DataFrame({c: data[c] for c in order})

    # -- introspection ----------------------------------------------------- #
    def tree(self):
        """Indented listing of every group and dataset with shape/dtype/units."""
        lines = [f"{self.path}"]

        def visit(name, obj):
            indent = "  " * (name.count("/") + 1)
            base = name.rsplit("/", 1)[-1]
            if isinstance(obj, h5py.Dataset):
                units = _py(obj.attrs.get("units", ""))
                tail = f"  [{units}]" if units else ""
                lines.append(f"{indent}{base}  {tuple(obj.shape)} "
                             f"{obj.dtype}{tail}")
            else:
                lines.append(f"{indent}{base}/")

        self._f.visititems(visit)
        return "\n".join(lines)

    def summary(self, file=None):
        """Print what this file contains: version, exposure, axes, cap, provenance.

        Everything printed is read from the file, so this is also the quickest
        way to see whether a cube you were sent is the one you expected.
        """
        out = []
        w = out.append
        w("=" * 78)
        w(f"POLONAISE UHDM data release   {self.version_tag}")
        w("=" * 78)
        w(f"file            : {self.path}")
        w(f"format          : {self.attrs.get('file_format')} "
          f"version {self.attrs.get('version')} "
          f"(schema {self.attrs.get('schema_version')})")
        w(f"created         : {self.attrs.get('created')}")
        exp = self.exposure_s
        w(f"exposure        : {exp:,.0f} s  ({exp / 3600:.2f} h)")
        cap = self.b_constrained_max_m
        w(f"impact-param cap: {'none (uncapped)' if cap is None else f'{cap} m'}")
        w(f"recommended CL  : {self.confidence_recommended}")

        w("")
        w("hypothesis axes")
        w(f"  f_dm          : {self.f_dm_values}  "
          f"(default {self._default_f_dm(None)}) -- DM fraction, pure flux scale")
        w(f"  atmosphere    : {[int(v) for v in self._axes['atmosphere']]} "
          f"-> {self.atmosphere_values}  (1/True = attenuation applied)")
        w(f"  mode          : {self.modes}")

        w("")
        w("axes")
        for name in sorted(self._axes):
            a = self._axes[name]
            at = self._axis_attrs.get(name, {})
            units = at.get("units", "")
            finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
            if a.size <= 6:
                rng = _fmt_list(a)
            else:
                rng = (f"{_fmt(np.min(finite))} .. {_fmt(np.max(finite))}"
                       if finite.size else "-")
            extra = ""
            if name == "lambda_m":
                n_inf = int(np.count_nonzero(~np.isfinite(a)))
                extra = (f"  [{self.n_finite_lambda} finite + {n_inf} massless "
                         f"sentinel(inf); tags: "
                         f"{', '.join(sorted(self.lambda_tags))}]")
            w(f"  {name:<15} n={a.size:<5} {rng:<28} [{units}]{extra}")
            desc = at.get("description")
            if desc:
                w(f"  {'':<15} {desc}")

        w("")
        w("results  (axis order in parentheses)")
        for q in self.quantities:
            ds = self._f["results"][q]
            w(f"  {q:<15} {str(tuple(ds.shape)):<26} {str(ds.dtype):<8} "
              f"[{_py(ds.attrs.get('units', ''))}]")
            w(f"  {'':<15} ({', '.join(self._dims_of(ds))})")
            w(f"  {'':<15} {_py(ds.attrs.get('description', ''))}")

        st = self._f["results"]["status"][:]
        total = st.size
        w("")
        w("status codes  (counts over the whole cube)")
        legend = self.status_legend()
        codes, counts = np.unique(st, return_counts=True)
        for c, n in zip(codes.tolist(), counts.tolist()):
            w(f"  {c}  {n:>12,}  ({100 * n / total:5.2f}%)  {legend.get(c, '?')}")

        w("")
        w("detector")
        w(f"  exposure_s     {exp:,.0f} s")
        for m in self.modes:
            ev = self.events(m)
            blips = self.all_blips(m)
            w(f"  mode {m}: {ev.size:>4} analysis events "
              f"({_fmt(ev.min())} .. {_fmt(ev.max())} GeV), "
              f"{blips.size} raw blips")
        w(f"  efficiency     q_gev_<mode>, eff_<mode>_df<2|3>; "
          f"analysis used df={self.attrs.get('df')}")

        if "halo" in self._f:
            w("")
            w("halo diagnostics (own coarser alpha/mass grids)")
            for name in sorted(self._f["halo"]):
                ds = self._f["halo"][name]
                w(f"  {name:<15} {str(tuple(ds.shape)):<26} "
                  f"[{_py(ds.attrs.get('units', ''))}]  "
                  f"{_py(ds.attrs.get('description', ''))}")

        if "reference_curves" in self._f:
            w("")
            w(f"reference_curves: {len(self._f['reference_curves'])} datasets "
              f"(showcase spectra / arrival-speed distributions)")

        w("")
        w("provenance")
        w(f"  git_commit     {self.attrs.get('git_commit')} "
          f"(dirty={self.attrs.get('git_dirty')})")
        w(f"  seed           {self.attrs.get('seed')}")
        w(f"  MC fidelity    n_mc={self.attrs.get('fid_n_mc')} "
          f"n_ode={self.attrs.get('fid_n_ode')} "
          f"n_shm={self.attrs.get('fid_n_shm')} n_q={self.attrs.get('fid_n_q')}")
        pkgs = self.attrs.get("packages_json")
        if pkgs:
            w(f"  packages       {pkgs}")
        for m in self.modes:
            key = f"events_mode{m}_sha256"
            if key in self.attrs:
                w(f"  {key:<14} {self.attrs[key]}")
        if "efficiency_npz_sha256" in self.attrs:
            w(f"  efficiency_npz_sha256  {self.attrs['efficiency_npz_sha256']}")
        w("=" * 78)

        text = "\n".join(out)
        print(text, file=file)
        return None


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    _PROG = sys.argv[0].replace("\\", "/").rsplit("/", 1)[-1]
    _USAGE = (
        f"usage: python {_PROG} <luhdm_datarelease_*.h5>\n"
        f"  Prints a summary of the release file: version tag, axes,\n"
        f"  quantities, status-code counts, detector inputs and provenance.\n"
        f"  For the API, import it instead: "
        f"import luhdm_release; help(luhdm_release)"
    )
    _args = sys.argv[1:]
    _asked_for_help = bool(_args) and _args[0] in ("-h", "--help", "help")

    if _asked_for_help or len(_args) != 1:
        print(__doc__)
        print(_USAGE)
        raise SystemExit(0 if _asked_for_help else 2)
    if _args[0].startswith("-"):
        print(f"unrecognised option {_args[0]!r}. This script takes exactly "
              f"one argument, the path to the release file.\n\n{_USAGE}",
              file=sys.stderr)
        raise SystemExit(2)
    try:
        _f = open_release(_args[0])
    except (FileNotFoundError, OSError) as _exc:
        print(f"could not open {_args[0]!r}: {_exc}\n\n"
              f"Pass the path to the release HDF5, for example "
              f"'luhdm_datarelease_v10_A_f1_atm.h5'. See README.md for where to get "
              f"it.\n\n{_USAGE}", file=sys.stderr)
        raise SystemExit(2) from None
    with _f as _rel:
        _rel.summary()
