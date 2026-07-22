"""Tests for luhdm.release: schema invariants, index resolution, plane reads,
the shared best-mass criterion, and error handling.

These build a small *synthetic* HDF5 file (:func:`make_mini_release`) that
follows the data-release contract exactly, so the loader is exercised without
the multi-hour real build.
"""
import json

import h5py
import numpy as np
import pytest

from luhdm import config, units
from luhdm import release
from luhdm.release import best_mass_index, open_release


# --------------------------------------------------------------------------- #
# Synthetic mini-release
# --------------------------------------------------------------------------- #
def make_mini_release(path, version=1):
    """Write a tiny but contract-faithful data-release file at ``path``.

    Small axes: 4 alphas, 5 atm masses, 6 noatm masses, lambda = three finite
    tag ranges (20um, 200um, 2mm) + inf, 3x3 halo grid. Cube values are chosen
    so orientation and specific loader behaviours are checkable.
    """
    alpha_n = np.geomspace(1e-8, 1.0, 4)
    mass_atm = np.geomspace(1e6, 1e18, 5)
    mass_noatm = np.geomspace(1e6, 1e18, 6)
    mass_halo = np.geomspace(1e6, 1e18, 3)
    alpha_halo = np.geomspace(2e-11, 1.0, 3)

    # finite ranges (ascending) that are exact tag members, then inf last.
    lam_finite = np.sort(np.array([2e-5, 2e-4, 2e-3]))
    lam_m = np.concatenate([lam_finite, [np.inf]])
    n_finite = lam_finite.size
    m_phi = np.array(
        [1.0 / units.conv_m2pGeV(x) for x in lam_finite] + [0.0])

    na, nm_a, nm_n, nl = alpha_n.size, mass_atm.size, mass_noatm.size, lam_m.size

    # -- atm cubes -------------------------------------------------------- #
    k = np.arange(3)[:, None, None, None]
    ia = np.arange(na)[None, :, None, None]
    im = np.arange(nm_a)[None, None, :, None]
    il = np.arange(nl)[None, None, None, :]
    # mu / n_transit: index encodings for orientation round-trip checks.
    mu_atm = (1000 * k + 100 * ia + 10 * im + il).astype(np.float32) \
        * np.ones((3, na, nm_a, nl), np.float32)
    ntr_atm = (100 * np.arange(na)[:, None, None]
               + 10 * np.arange(nm_a)[None, :, None]
               + np.arange(nl)[None, None, :]).astype(np.float32)
    status_atm = (np.arange(nl)[None, None, None, :]
                  * np.ones((3, na, nm_a, nl))).astype(np.uint8)
    # extremeness: distinct per-mode base (mode 2 strictly largest) so the
    # composite (elementwise max over modes) is predictable.
    ext_atm = (0.1 + 0.25 * k + 0.001 * (ia + im + il)) \
        * np.ones((3, na, nm_a, nl), np.float32)
    ext_atm = ext_atm.astype(np.float32)
    # a hand-built exclusion band for mode 1 (k=0), mass idx 2, lambda 200um
    # (il=1): p(alpha) rises above then falls below 0.95.
    ext_atm[0, :, 2, 1] = np.array([0.2, 0.97, 0.98, 0.3], np.float32)
    # a known f8 value for the float32 round-trip test (mode1, ia0, im0, inf).
    ext_atm[0, 0, 0, 3] = np.float32(0.9498765)

    # -- noatm cubes (own mass axis) ------------------------------------- #
    imn = np.arange(nm_n)[None, None, :, None]
    ext_noatm = (0.1 + 0.25 * k + 0.001 * (ia + imn + il)) \
        * np.ones((3, na, nm_n, nl), np.float32)
    mu_noatm = (1000 * k + 100 * ia + 10 * imn + il).astype(np.float32) \
        * np.ones((3, na, nm_n, nl), np.float32)
    ntr_noatm = (100 * np.arange(na)[:, None, None]
                 + 10 * np.arange(nm_n)[None, :, None]
                 + np.arange(nl)[None, None, :]).astype(np.float32)
    status_noatm = (np.arange(nl)[None, None, None, :]
                    * np.ones((3, na, nm_n, nl))).astype(np.uint8)

    # -- halo (alpha_halo, mass_halo, lambda) ---------------------------- #
    nth = (np.arange(3)[:, None, None] * 1.0 + np.arange(3)[None, :, None]
           + np.arange(nl)[None, None, :]).astype(np.float32)
    bmax = (config.R_EFF + 1e-6 * (np.arange(3)[:, None, None]
            + np.arange(3)[None, :, None]
            + np.arange(nl)[None, None, :])).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["file_format"] = "luhdm-datarelease"
        f.attrs["version"] = version
        f.attrs["schema_version"] = 1
        f.attrs["created"] = "2026-07-19T00:00:00Z"
        f.attrs["seed"] = 20260702
        f.attrs["q_thresh_gev"] = config.Q_THRESH
        f.attrs["t_exposure_s"] = config.T_EXPOSURE
        f.attrs["m_planck_gev"] = 1.22e19
        f.attrs["confidence_recommended"] = 0.95
        f.attrs["q_hi_ref_gev"] = 8400.0

        ax = f.create_group("axes")
        d = ax.create_dataset("mode", data=np.array([1, 2, 3], np.uint8))
        d.attrs["units"] = "index"
        ax.create_dataset("alpha_n", data=alpha_n).attrs["units"] = "dimensionless"
        ax.create_dataset("mass_gev", data=mass_atm).attrs["units"] = "GeV"
        ax.create_dataset("mass_noatm_gev", data=mass_noatm).attrs["units"] = "GeV"
        dl = ax.create_dataset("lambda_m", data=lam_m)
        dl.attrs["units"] = "m"
        dl.attrs["n_finite"] = n_finite
        dl.attrs["tags_json"] = json.dumps(
            {t: release.TAGS[t] for t in release.TAG_ORDER})
        ax.create_dataset("m_phi_gev", data=m_phi).attrs["units"] = "GeV"
        ax.create_dataset("mass_halo_gev", data=mass_halo).attrs["units"] = "GeV"
        ax.create_dataset("alpha_halo_n", data=alpha_halo).attrs["units"] = \
            "dimensionless"

        atm = f.create_group("atm")
        atm.create_dataset("extremeness", data=ext_atm).attrs["units"] = "prob"
        atm.create_dataset("mu", data=mu_atm).attrs["units"] = "counts"
        atm.create_dataset("n_transit", data=ntr_atm).attrs["units"] = "counts"
        atm.create_dataset("status", data=status_atm).attrs["units"] = "code"

        noatm = f.create_group("noatm")
        noatm.create_dataset("extremeness", data=ext_noatm).attrs["units"] = "prob"
        noatm.create_dataset("mu", data=mu_noatm).attrs["units"] = "counts"
        noatm.create_dataset("n_transit", data=ntr_noatm).attrs["units"] = "counts"
        noatm.create_dataset("status", data=status_noatm).attrs["units"] = "code"

        halo = f.create_group("halo")
        halo.create_dataset("n_transit", data=nth).attrs["units"] = "counts"
        halo.create_dataset("bmax", data=bmax).attrs["units"] = "m"

        det = f.create_group("detector")
        det.create_dataset("exposure_s", data=config.T_EXPOSURE)
        for mode in (1, 2, 3):
            det.create_dataset(f"events_mode{mode}",
                               data=np.array([1.5e2 * mode]))
            det.create_dataset(f"all_blips_mode{mode}",
                               data=np.array([1e11, 2e11]) * mode)
            det.create_dataset(f"q_gev_{mode}", data=np.geomspace(1e2, 1e5, 5))
            det.create_dataset(f"eff_{mode}_df2", data=np.linspace(0, 1, 5))
            det.create_dataset(f"eff_{mode}_df3", data=np.linspace(0, 0.9, 5))

        rc = f.create_group("reference_curves")
        rc.create_dataset("v", data=np.linspace(0.0, config.VESC, 7))
        for i, tag in enumerate(("shm",) + tuple(release.TAG_ORDER)):
            ds = rc.create_dataset(f"fv_{tag}",
                                   data=np.linspace(0, 1, 7).astype(np.float32))
            ds.attrs["survival_fraction"] = 1.0 if tag == "shm" else 0.5 + 0.05 * i
        rc.create_dataset("q_gev", data=np.geomspace(1e2, 1e5, 6))
        for tag in tuple(release.TAG_ORDER) + ("massless",):
            rc.create_dataset(f"drdq_{tag}",
                              data=np.linspace(1, 0, 6).astype(np.float32))
    return path


@pytest.fixture
def rel(tmp_path):
    p = make_mini_release(tmp_path / "mini.h5")
    r = open_release(p)
    yield r
    r.close()


# --------------------------------------------------------------------------- #
# Schema invariants
# --------------------------------------------------------------------------- #
def test_schema_invariants(rel):
    ax = rel.axes
    assert np.isinf(ax.lambda_m[-1])            # massless sentinel last
    assert ax.m_phi_gev[-1] == 0.0              # exactly zero for massless
    # monotone axes
    assert np.all(np.diff(ax.mass_gev) > 0)
    assert np.all(np.diff(ax.alpha_n) > 0)
    assert np.all(np.diff(ax.lambda_finite) > 0)
    assert np.all(np.diff(rel.axes_noatm.mass_gev) > 0)
    assert ax.n_finite == len(ax.lambda_finite)
    # finite m_phi is 1/conv(lambda)
    assert np.isclose(ax.m_phi_gev[0], 1.0 / units.conv_m2pGeV(ax.lambda_m[0]))


def test_tags_are_exact_axis_members(rel):
    lam = rel.axes.lambda_m
    for tag in ("20um", "200um", "2mm"):
        assert np.any(lam == release.TAGS[tag]), f"{tag} not an exact axis member"
    assert np.any(np.isinf(lam))                # 'massless'


# --------------------------------------------------------------------------- #
# Index resolution
# --------------------------------------------------------------------------- #
def test_at_lambda_exact(rel):
    assert rel.at_lambda("200um") == 1
    assert rel.at_lambda(2e-4) == 1
    assert rel.at_lambda("20um") == 0
    assert rel.at_lambda("2mm") == 2
    assert rel.at_lambda(np.inf) == 3
    assert rel.at_lambda("massless") == 3


def test_at_lambda_miss_lists_nearest(rel):
    with pytest.raises(KeyError) as exc:
        rel.at_lambda(1.5e-4)          # between 20um and 200um, not on axis
    msg = str(exc.value)
    assert "nearest" in msg
    # the true nearest finite value (2e-4) should be among the reported three
    assert any(np.isclose(2e-4, v, rtol=1e-6)
               for v in _floats_in(msg))


def _floats_in(s):
    import re
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def test_at_mass_and_alpha_nearest(rel):
    ax = rel.axes
    # exact hits
    assert rel.at_mass(ax.mass_gev[2]) == 2
    assert rel.at_alpha(ax.alpha_n[1]) == 1
    # a value just off a grid point snaps to nearest in log10
    assert rel.at_mass(ax.mass_gev[3] * 1.01) == 3
    # halo group uses the halo axes
    assert rel.at_mass(rel.axes_halo.mass_gev[1], group="halo") == 1
    assert rel.at_alpha(rel.axes_halo.alpha_n[2], group="halo") == 2


# --------------------------------------------------------------------------- #
# Plane reads / orientation
# --------------------------------------------------------------------------- #
def test_mass_plane_orientation_mu(rel):
    # mu was encoded as 1000*k + 100*ia + 10*im + il
    plane = rel.mass_plane("mu", mode=2, lam="2mm", group="atm")  # k=1, il=2
    na, nm = rel.axes.alpha_n.size, rel.axes.mass_gev.size
    assert plane.shape == (na, nm)
    expected = np.array([[1000 * 1 + 100 * a + 10 * m + 2
                          for m in range(nm)] for a in range(na)])
    np.testing.assert_allclose(plane, expected)


def test_mass_plane_n_transit_is_modeless(rel):
    plane = rel.mass_plane("n_transit", lam="20um", group="atm")  # il=0
    na, nm = rel.axes.alpha_n.size, rel.axes.mass_gev.size
    expected = np.array([[100 * a + 10 * m + 0 for m in range(nm)]
                         for a in range(na)])
    np.testing.assert_allclose(plane, expected)
    # requesting a mode for a mode-less quantity is an error
    with pytest.raises(ValueError):
        rel.mass_plane("n_transit", mode=1, lam="20um")
    # requesting extremeness without a mode is an error
    with pytest.raises(ValueError):
        rel.mass_plane("extremeness", lam="20um")


def test_mass_plane_halo(rel):
    plane = rel.mass_plane("n_transit", lam="200um", group="halo")  # il=1
    assert plane.shape == (rel.axes_halo.alpha_n.size,
                           rel.axes_halo.mass_gev.size)
    # halo cubes are mode-less
    with pytest.raises(ValueError):
        rel.mass_plane("n_transit", mode=1, lam="200um", group="halo")
    with pytest.raises(ValueError):
        rel.mass_plane("extremeness", lam="200um", group="halo")


def test_float32_roundtrip(rel):
    plane = rel.mass_plane("extremeness", mode=1, lam="massless", group="atm")
    assert plane[0, 0] == pytest.approx(0.9498765, abs=1e-6)


def test_lambda_plane_shape(rel):
    lp = rel.lambda_plane("extremeness", mode=1, mass=rel.axes.mass_gev[2])
    assert lp.shape == (rel.axes.alpha_n.size, rel.axes.n_finite)
    # mass=None routes through best_mass without error
    lp2 = rel.lambda_plane("mu", mode=1)
    assert lp2.shape == (rel.axes.alpha_n.size, rel.axes.n_finite)


# --------------------------------------------------------------------------- #
# Composite = elementwise max over modes
# --------------------------------------------------------------------------- #
def test_composite_is_elementwise_max(rel):
    comp = rel.composite("extremeness", lam="20um", group="atm")
    planes = [rel.mass_plane("extremeness", mode=m, lam="20um", group="atm")
              for m in (1, 2, 3)]
    np.testing.assert_allclose(comp, np.maximum.reduce(planes))
    # and it truly differs from any single mode here (mode 2 is largest)
    np.testing.assert_allclose(comp, planes[2])
    with pytest.raises(ValueError):
        rel.composite("n_transit", lam="20um")   # not a per-mode quantity


# --------------------------------------------------------------------------- #
# best_mass_index criterion (pure function)
# --------------------------------------------------------------------------- #
def test_best_mass_index_shortest_lambda_restriction():
    """The shortest-excluded-lambda restriction must override raw area.

    Mass 0 has by far the largest excluded area but only reaches the two
    *longer* lambdas; mass 1 reaches the shortest lambda with a tiny area;
    mass 2 excludes nothing. The criterion must pick mass 1 (candidates are
    restricted to masses reaching the globally shortest excluded lambda),
    even though a pure argmax-of-area would pick mass 0.
    """
    alpha_n = np.geomspace(1e-8, 1.0, 4)
    lam_finite = np.array([1e-3, 1e-2, 1e-1])   # ascending
    conf = 0.95
    na, nmass, nl = 4, 3, 3
    p = np.zeros((na, nmass, nl))
    # mass 0: all alphas excluded at the two longer lambdas (il=1,2); none at il=0
    p[:, 0, 1] = 1.0
    p[:, 0, 2] = 1.0
    # mass 1: one alpha excluded at the shortest lambda (il=0) only
    p[0, 1, 0] = 1.0
    # mass 2: nothing

    # sanity: pure argmax of the area would choose mass 0
    dloga = np.mean(np.diff(np.log10(alpha_n)))
    W = (p >= conf).sum(axis=0).T * dloga
    area = np.trapezoid(W, np.log10(lam_finite), axis=0)
    assert int(np.argmax(area)) == 0

    # the real criterion restricts to shortest-lambda reachers -> mass 1
    assert best_mass_index(p, lam_finite, alpha_n, conf) == 1


def test_best_mass_index_matches_notebook_cell(rel):
    """best_mass_index reproduces notebook 01 cell-12's expression exactly."""
    # build a random-ish but reproducible per-mode extremeness cube on the file
    imode = 0
    nfin = rel.axes.n_finite
    p_finite = rel._file["atm"]["extremeness"][imode, :, :, :nfin]
    conf = 0.95
    got = best_mass_index(p_finite, rel.axes.lambda_finite, rel.axes.alpha_n, conf)

    # replicate the notebook's cell-12 logic verbatim (SCANS-style, per tag)
    _al = rel.axes.alpha_n
    _ms = rel.axes.mass_gev
    _dloga = np.diff(np.log10(_al)).mean()
    lam_fin = rel.axes.lambda_finite
    # SCANS[t]["extremeness"] is (n_alpha, n_mass); here the lambda index is t
    _W = np.array([(p_finite[:, :, i] >= conf).sum(axis=0) * _dloga
                   for i in range(lam_fin.size)])             # (range, mass)
    _loglam = np.log10(lam_fin)
    _ord = np.argsort(_loglam)
    _area = np.trapezoid(_W[_ord], _loglam[_ord], axis=0)
    _shortest = np.array([
        min([lam_fin[ri] for ri in range(lam_fin.size) if _W[ri, m] > 0],
            default=np.inf) for m in range(_ms.size)])
    _cand = np.where(_shortest == _shortest.min())[0]
    m_best_nb = int(_cand[np.argmax(_area[_cand])])
    assert got == m_best_nb


def test_best_mass_memoised(rel):
    m1 = rel.best_mass(1)
    m2 = rel.best_mass(1)
    assert m1 == m2
    assert m1[0] == rel.axes.mass_gev[m1[1]]
    assert isinstance(m1[1], int)


# --------------------------------------------------------------------------- #
# excluded_alpha_band
# --------------------------------------------------------------------------- #
def test_excluded_alpha_band(rel):
    # column ext[mode1, :, mass2, 200um] = [0.2, 0.97, 0.98, 0.3] -> finite band
    lo, hi = rel.excluded_alpha_band(rel.axes.mass_gev[2], "200um", mode=1)
    a = rel.axes.alpha_n
    assert np.isfinite(lo) and np.isfinite(hi)
    assert a[0] < lo < a[1]
    assert a[2] < hi < a[3]


def test_excluded_alpha_band_nothing(rel):
    # mode 1 base extremeness (~0.1) never crosses 0.95 -> (nan, nan)
    lo, hi = rel.excluded_alpha_band(rel.axes.mass_gev[0], "20um", mode=1)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# cell() dump
# --------------------------------------------------------------------------- #
def test_cell_dump(rel):
    c = rel.cell(rel.axes.mass_gev[2], rel.axes.alpha_n[1], "2mm", mode=2)
    for key in ("mass_gev", "alpha_n", "lambda_m", "indices", "extremeness",
                "mu", "n_transit", "status", "bmax", "n_transit_halo"):
        assert key in c
    ia, im, il = c["indices"]
    assert (ia, im, il) == (1, 2, 2)
    # mu index encoding: 1000*k + 100*ia + 10*im + il, k=1
    assert c["mu"] == pytest.approx(1000 * 1 + 100 * 1 + 10 * 2 + 2)
    assert c["lambda_m"] == pytest.approx(2e-3)
    assert c["bmax"] >= config.R_EFF


# --------------------------------------------------------------------------- #
# detector / reference curves
# --------------------------------------------------------------------------- #
def test_detector_accessors(rel):
    assert rel.events(1).shape == (1,)
    assert rel.all_blips(2).shape == (2,)
    q, e = rel.efficiency_curve(3, df=3)
    assert q.shape == e.shape == (5,)
    q2, e2 = rel.efficiency_curve(3, df=2)
    assert not np.allclose(e, e2)      # df selects a different curve


def test_reference_curves(rel):
    v, fv = rel.speed_dist("200um")
    assert v.shape == fv.shape == (7,)
    assert rel.survival_fraction("shm") == pytest.approx(1.0)
    assert 0 < rel.survival_fraction("2mm") < 1
    q, dr = rel.raw_spectrum("massless")
    assert q.shape == dr.shape == (6,)
    with pytest.raises(ValueError):
        rel.speed_dist("massless")     # massless has no speed distribution
    with pytest.raises(ValueError):
        rel.raw_spectrum("shm")        # shm has no raw spectrum entry


def test_tree(rel):
    t = rel.tree()
    assert "extremeness" in t
    assert "lambda_m" in t
    assert "[m]" in t                  # units surfaced


# --------------------------------------------------------------------------- #
# Errors: version mismatch, missing file, context manager
# --------------------------------------------------------------------------- #
def test_version_mismatch(tmp_path):
    p = make_mini_release(tmp_path / "bad.h5", version=2)
    with pytest.raises(ValueError) as exc:
        open_release(p)
    msg = str(exc.value)
    assert "2" in msg and str(release.FORMAT_VERSION) in msg


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        open_release(tmp_path / "does_not_exist.h5")
    msg = str(exc.value)
    assert "README" in msg and "build_release" in msg


def test_context_manager(tmp_path):
    p = make_mini_release(tmp_path / "ctx.h5")
    with open_release(p) as r:
        assert r.axes.mass_gev.size == 5
    # file is closed on exit
    assert not r._file.id.valid
