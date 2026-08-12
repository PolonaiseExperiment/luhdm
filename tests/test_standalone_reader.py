"""The standalone release reader must stay faithful to ``luhdm.release``.

``release/luhdm_release.py`` is shipped to outside users who cannot install
this package, so it has to (a) never import luhdm and (b) return exactly what
the package loader returns. These tests are schema-driven: every axis value,
length and tag comes from whichever cube is present, so they keep working when
the cube is rebuilt with a different grid.
"""
import ast
import importlib.util
import itertools
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
READER = REPO / "release" / "luhdm_release.py"
CUBES = sorted((REPO / "release").glob("luhdm_datarelease_v*.h5"))


def _load_standalone():
    """Import release/luhdm_release.py by path, as an outside user would."""
    spec = importlib.util.spec_from_file_location("_standalone_reader", READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def std():
    return _load_standalone()


@pytest.fixture(scope="module")
def cube():
    """The newest release cube that the axis layout reader understands."""
    import h5py

    for path in reversed(CUBES):
        with h5py.File(path, "r") as f:
            if "results" in f and "axes" in f:
                return path
    pytest.skip("no axis-layout data release found under release/")


def test_reader_imports_nothing_from_luhdm():
    """Static check: no luhdm / optimum_interval import, no relative import."""
    tree = ast.parse(READER.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "standalone reader must not use relative imports"
            if node.module:
                mods.add(node.module.split(".")[0])
    assert not mods & {"luhdm", "optimum_interval"}
    assert mods <= {"__future__", "json", "sys", "warnings", "numpy", "h5py", "pandas"}


def test_reader_runs_with_luhdm_blocked(cube, tmp_path):
    """Subprocess check: import + use the reader with luhdm unimportable."""
    script = tmp_path / "check.py"
    script.write_text(f"""
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("luhdm", "optimum_interval"):
            raise ImportError("blocked: " + name)

sys.path[:] = [p for p in sys.path if p.rstrip("/") != {str(REPO)!r} and p not in ("", ".")]
sys.meta_path.insert(0, Block())
sys.path.insert(0, {str(READER.parent)!r})
import luhdm_release
with luhdm_release.open_release({str(cube)!r}) as rel:
    rel.summary()
    # the hypothesis comes from the cube: a v7 file carries exactly one, and it
    # need not be the reader's f_dm_default / atmosphere=True fallback
    rel.excluded_band(mode=rel.modes[0], lam="massless", nan_policy="ignore",
                      f_dm=rel.f_dm_values[0],
                      atmosphere=rel.atmosphere_values[0])
assert not [m for m in sys.modules if m.split(".")[0] in ("luhdm", "optimum_interval")]
print("OK")
""")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("OK")


def _points(rel_s):
    """A varied sample: every mode x f_dm x atmosphere, several lambdas.

    ``lambda_tags`` is inherited from the parent scan and can name ranges this
    cube does not carry, so tags are filtered against the axis before use.
    """
    lam_axis = rel_s.axis("lambda_m")
    n_fin = rel_s.n_finite_lambda
    on_axis = {t for t in rel_s.lambda_tags
               if t == "massless" or np.isin(rel_s.lambda_tags[t], lam_axis)}
    lams = ["massless", *sorted(t for t in on_axis if t != "massless")]
    lams += [float(lam_axis[i]) for i in (0, n_fin // 3, n_fin - 1)]
    return list(itertools.product(rel_s.modes, rel_s.f_dm_values,
                                  rel_s.atmosphere_values, lams))


def test_planes_match_package_loader(std, cube):
    """Every quantity, bit-identical, over modes x f_dm x atmosphere x lambda."""
    from luhdm import release as pkg

    rel_p = pkg.open_release(cube)
    rel_s = std.open_release(cube)
    try:
        points = _points(rel_s)
        # one hypothesis per file in the v7 layout, several in older cubes
        assert len(points) >= 3 * len(rel_s.modes)
        for mode, f_dm, atm, lam in points:
            group = "atm" if atm else "noatm"
            for q in ("extremeness", "mu", "status"):
                a = rel_p.mass_plane(q, mode=mode, lam=lam, group=group, f_dm=f_dm)
                b = rel_s.get(q, mode=mode, lam=lam, atmosphere=atm, f_dm=f_dm).values
                assert a.dtype == b.dtype
                assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
            a = rel_p.mass_plane("n_transit", lam=lam, group=group, f_dm=f_dm)
            b = rel_s.get("n_transit", lam=lam, atmosphere=atm, f_dm=f_dm).values
            assert np.array_equal(a, b, equal_nan=True)
    finally:
        rel_p.close()
        rel_s.close()


def test_excluded_band_matches_limits(std, cube):
    """The reimplemented boundary equals luhdm.limits.excluded_band exactly."""
    from luhdm import limits, release as pkg

    rel_p = pkg.open_release(cube)
    rel_s = std.open_release(cube)
    try:
        alpha = rel_p.axes.alpha_n
        mass = rel_p.axes.mass_gev
        for mode, f_dm, atm, lam in _points(rel_s)[::5]:
            group = "atm" if atm else "noatm"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                band = rel_s.excluded_band(mode=mode, lam=lam, f_dm=f_dm,
                                           atmosphere=atm)
            ext = rel_p.mass_plane("extremeness", mode=mode, lam=lam,
                                   group=group, f_dm=f_dm)
            ref = np.array([limits.excluded_band(alpha, ext[:, im], level=0.95)
                            for im in range(mass.size)])
            assert np.array_equal(ref[:, 0], band.alpha_lo, equal_nan=True)
            assert np.array_equal(ref[:, 1], band.alpha_hi, equal_nan=True)
    finally:
        rel_p.close()
        rel_s.close()


def test_nan_cells_are_surfaced_not_silent(std, cube):
    """status==1 cells read as 'not excluded' but are always reported."""
    rel = std.open_release(cube)
    try:
        st = rel.get("status").values
        ext = rel.get("extremeness").values
        assert np.array_equal(np.isnan(ext), st == 1)
        # find a hypothesis with failures and check it is warned about
        idx = np.argwhere(st == 1)
        if not idx.size:
            pytest.skip("cube has no status==1 cells")
        i_f, i_atm, i_mode, _, _, i_lam = idx[0]
        kw = dict(mode=int(rel.axis("mode")[i_mode]),
                  lam=float(rel.axis("lambda_m")[i_lam]),
                  f_dm=float(rel.axis("f_dm")[i_f]),
                  atmosphere=bool(rel.axis("atmosphere")[i_atm]))
        with pytest.warns(RuntimeWarning, match="NaN"):
            band = rel.excluded_band(**kw)
        assert band.n_undefined.sum() > 0
        with pytest.raises(ValueError, match="NaN"):
            rel.excluded_band(nan_policy="raise", **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rel.excluded_band(nan_policy="ignore", **kw)
    finally:
        rel.close()


def test_selection_errors_list_available_values(std, cube):
    """A missed selection must tell the user what the file actually carries."""
    rel = std.open_release(cube)
    try:
        with pytest.raises(KeyError, match="Nearest axis values"):
            rel.get("extremeness", mode=rel.modes[0], lam=3.7e-5)
        with pytest.raises(KeyError, match="unknown lambda tag"):
            rel.get("extremeness", mode=rel.modes[0], lam="banana")
        with pytest.raises(KeyError, match="not on the axis"):
            rel.get("extremeness", mode=max(rel.modes) + 99, lam="massless")
        with pytest.raises(ValueError, match="no 'mode' axis"):
            rel.get("n_transit", mode=rel.modes[0], lam="massless")
    finally:
        rel.close()


def test_massless_slice_is_the_infinite_lambda(std, cube):
    rel = std.open_release(cube)
    try:
        lam = rel.axis("lambda_m")
        i = int(np.flatnonzero(~np.isfinite(lam))[0])
        assert rel.n_finite_lambda == i
        assert float(rel.axis("m_phi_gev")[i]) == 0.0
        by_tag = rel.get("extremeness", mode=rel.modes[0], lam="massless")
        by_inf = rel.get("extremeness", mode=rel.modes[0], lam=np.inf)
        assert by_tag.indices["lambda_m"] == by_inf.indices["lambda_m"] == i
        assert np.array_equal(by_tag.values, by_inf.values, equal_nan=True)
    finally:
        rel.close()
