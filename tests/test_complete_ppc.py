"""Unit tests for `completion_rates`, the module-level function #510 adds to
replace the quadratic `complete_ppc` closure that used to live inside
`PPPrunStart`.

`_reference` below is that closure's pre-#510 body, copied verbatim from
`PPPrunStart.complete_ppc` (`git show dev-main:src/pfs_target_uploader/utils/ppp.py`,
lines 1414-1500), turned into a module-level function by making
`single_exptime` an explicit parameter instead of a closure variable, and
dropping the `logger.info(...)` call (there is no logger in scope here). It is
the oracle these tests compare `completion_rates` against, so it must not be
"improved" to look more like the new implementation -- any divergence from the
new code should show up as a test failure here, not be quietly patched away.
"""

import math

import numpy as np
import pytest
from astropy.table import Column, Table

from pfs_target_uploader.utils.ppp import completion_rates

# PPPrunStart's own default (`single_exptime: int = 900`), and what production
# actually passes. Passing a float here instead would make `_reference` raise a
# casting error on the very first non-empty pointing: `exptime_assign` starts
# out as an int64 column (from the unconditional `sample["exptime_assign"] =
# 0` at the top of the closure) and `.data[lst] += single_exptime` adds into
# it in place *before* the first `np.minimum` call promotes the column to
# float64; numpy's "same_kind" in-place casting rejects float -> int64.
SINGLE_EXPTIME = 900


def _reference(sample, point_l, single_exptime):
    """Pre-#510 `complete_ppc`, kept byte-for-byte as the oracle for
    `completion_rates` (see module docstring). Do not "clean up" this
    function -- its builtin `sum()` calls and full-table slicing per pointing
    are exactly what #510 replaces, and quietly improving it here would hide
    a regression instead of catching one.
    """
    sample["exptime_assign"] = 0
    sub_l = sorted(list(set(sample["priority"])))  # noqa: C414 -- verbatim, see above
    n_sub = len(sub_l)

    if len(point_l) == 0:
        return (
            sample,
            np.array([[0] * (n_sub + 1)]),
            np.array([[0] * (n_sub + 1)]),
            np.array([[0] * (n_sub + 1)]),
            np.array([[0] * (n_sub + 1)]),
            sub_l,
        )

    point_l_pri = point_l[
        point_l.argsort(keys="ppc_priority")
    ]  # sort ppc by its total priority == sum(weights of the assigned targets in ppc)

    # sub-groups of the input sample, catagarized by the user defined priority
    count_sub_fh = [sum(sample["exptime"]) / 3600.0] + [
        sum(sample[sample["priority"] == ll]["exptime"]) / 3600.0 for ll in sub_l
    ]  # fiber hours
    count_sub_n = [len(sample)] + [
        sum(sample["priority"] == ll) for ll in sub_l
    ]  # number count of complete targets

    completeR_fh = []  # fiber hours
    completeR_fh_ = []  # percentage

    completeR_n = []  # number count of complete targets
    completeR_n_ = []  # percentage

    for ppc in point_l_pri:
        lst = np.where(np.isin(sample["ob_code"], ppc["allocated_targets"]))[0]
        sample["exptime_assign"].data[lst] += single_exptime
        sample["exptime_assign"] = np.minimum(
            sample["exptime_assign"], sample["exptime"]
        )

        # achieved fiber hours (in total, in P[0-9])
        comT_t_fh = [sum(sample["exptime_assign"]) / 3600.0] + [
            sum(sample[sample["priority"] == ll]["exptime_assign"]) / 3600.0
            for ll in sub_l
        ]

        comp_s = np.where(sample["exptime"] <= sample["exptime_assign"])[0]
        comT_t_n = [len(comp_s)] + [
            sum(sample["priority"].data[comp_s] == ll) for ll in sub_l
        ]

        completeR_fh.append(comT_t_fh)
        completeR_fh_.append(
            [comT_t_fh[oo] / count_sub_fh[oo] * 100 for oo in range(len(count_sub_fh))]
        )

        completeR_n.append(comT_t_n)
        completeR_n_.append(
            [comT_t_n[oo] / count_sub_n[oo] * 100 for oo in range(len(count_sub_n))]
        )

    return (
        sample,
        np.array(completeR_fh),
        np.array(completeR_fh_),
        np.array(completeR_n),
        np.array(completeR_n_),
        sub_l,
    )


class _FakeLogger:
    """Minimal stand-in for the loguru logger `completion_rates` accepts.
    Records `.info()` calls instead of printing them, so a test can check
    what was logged without pulling in loguru's own capture machinery.
    """

    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _make_case(n_target, n_point, n_group, priority_dtype, single_exptime, seed):
    """Build a seeded synthetic (sample, point_l) pair.

    `sample` has unique `ob_code` (numpy `<U` strings), `priority` drawn from
    `[0, n_group)` with the requested dtype, `exptime` mixing multiples and
    non-multiples of `single_exptime`, and dummy `ra`/`dec` columns.

    `point_l` has `ppc_priority` (float, rounded so ties are likely) and
    `allocated_targets` (an object column of lists of str/np.str_). When
    there is room (`n_point >= 1` / `>= 2`), the allocations are seeded to
    always exercise: a duplicated code within one pointing, unknown codes not
    present in `sample`, an empty allocation list, and a couple of "hot"
    targets allocated in more pointings than `ceil(exptime /
    single_exptime)` needs. The rest of each pointing's allocation is drawn
    at random for realism (`seed` makes the whole thing reproducible).
    """
    rng = np.random.default_rng(seed)

    codes = np.array([f"ob_{i:06d}" for i in range(n_target)])
    exptime_options = np.array([450.0, 900.0, 1800.0, 1234.5, 3600.0])
    exptime = rng.choice(exptime_options, size=n_target)

    raw_priority = rng.integers(0, n_group, size=n_target)
    priority = raw_priority.astype(np.float64 if priority_dtype is float else np.int64)

    ra = rng.uniform(0.0, 360.0, size=n_target)
    dec = rng.uniform(-40.0, 90.0, size=n_target)

    sample = Table(
        {
            "ob_code": codes,
            "priority": priority,
            "exptime": exptime,
            "ra": ra,
            "dec": dec,
        }
    )

    allocated = []
    for _ in range(n_point):
        k = int(rng.integers(0, n_target + 1))
        chosen = rng.choice(codes, size=k, replace=True) if k > 0 else codes[:0]
        allocated.append(list(chosen))

    if n_point >= 1:
        # Pointing 0: a duplicated code plus two codes absent from `sample`.
        allocated[0] = allocated[0] + [
            codes[0],
            codes[0],
            "unknown_1",
            "zz_not_in_sample",
        ]

    if n_point >= 2:
        # Last pointing: always empty.
        allocated[-1] = []

        # A couple of "hot" targets, over-allocated past what they need to
        # reach `exptime`. Drawn from every pointing except the forced-empty
        # last one, so that one reliably stays empty.
        n_hot = min(2, n_target)
        hot_idx = rng.choice(n_target, size=n_hot, replace=False)
        candidates = np.arange(n_point - 1)
        for i in hot_idx:
            code = codes[i]
            needed = math.ceil(exptime[i] / single_exptime)
            n_extra = min(len(candidates), needed + 5)
            for p in rng.choice(candidates, size=n_extra, replace=False):
                allocated[p].append(code)

    ppc_priority = rng.uniform(0.0, 1.0, size=n_point)
    if n_point > 0:
        ppc_priority = np.round(ppc_priority, 1)  # encourage ties

    point_l = Table()
    point_l["ppc_priority"] = Column(np.asarray(ppc_priority, dtype=np.float64))
    point_l["allocated_targets"] = Column(allocated, dtype=object)

    return sample, point_l


# ---------------------------------------------------------------------------
# Oracle comparison: completion_rates() must reproduce _reference() exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_target", [1, 7, 250], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("n_point", [0, 1, 5, 60], ids=lambda p: f"P{p}")
@pytest.mark.parametrize("n_group", [1, 3, 10], ids=lambda g: f"G{g}")
@pytest.mark.parametrize(
    "priority_dtype", [int, float], ids=["int-priority", "float-priority"]
)
def test_matches_pre_510_oracle(n_target, n_point, n_group, priority_dtype):
    """`completion_rates` must reproduce the pre-#510 closure exactly: same
    assigned exptime (value and dtype), same `sub_l` (value and element
    type), and curves equal up to float rounding, on the same input.
    """
    seed = (n_target, n_point, n_group, 0 if priority_dtype is int else 1)
    base_sample, point_l = _make_case(
        n_target, n_point, n_group, priority_dtype, SINGLE_EXPTIME, seed
    )

    old = _reference(base_sample.copy(), point_l, SINGLE_EXPTIME)
    new = completion_rates(base_sample.copy(), point_l, SINGLE_EXPTIME)

    old_assign = np.asarray(old[0]["exptime_assign"])
    new_assign = np.asarray(new[0]["exptime_assign"])
    assert np.array_equal(old_assign, new_assign)
    assert old_assign.dtype == new_assign.dtype

    assert old[5] == new[5]
    assert all(type(o) is type(n) for o, n in zip(old[5], new[5]))

    n_sub = len(old[5])
    expected_shape = (n_point if n_point > 0 else 1, n_sub + 1)

    for idx in (1, 2, 4):  # fh, fh_pct, n_pct
        assert old[idx].shape == expected_shape
        assert new[idx].shape == expected_shape
        assert np.allclose(old[idx], new[idx], rtol=0, atol=1e-9)

    assert old[3].shape == expected_shape
    assert new[3].shape == expected_shape
    assert np.array_equal(old[3], new[3])
    assert old[3].dtype == np.int64
    assert new[3].dtype == np.int64


# ---------------------------------------------------------------------------
# Contract tests: properties completion_rates() must have on its own, not
# just agreement with the oracle.
# ---------------------------------------------------------------------------


def test_returns_same_table_object_and_adds_column():
    """Element 0 of the return value must be the input table itself, not a
    copy -- callers reuse the input table after the call."""
    sample, point_l = _make_case(10, 4, 2, int, SINGLE_EXPTIME, seed=(10, 4, 2, 101))
    assert "exptime_assign" not in sample.colnames

    result = completion_rates(sample, point_l, SINGLE_EXPTIME)

    assert result[0] is sample
    assert "exptime_assign" in sample.colnames


def test_returns_same_table_object_when_no_pointings():
    sample, point_l = _make_case(10, 0, 2, int, SINGLE_EXPTIME, seed=(10, 0, 2, 102))

    result = completion_rates(sample, point_l, SINGLE_EXPTIME)

    assert result[0] is sample
    assert "exptime_assign" in sample.colnames


def test_zero_pointings_returns_all_zero_curves_and_logs():
    n_target, n_group = 6, 3
    sample, point_l = _make_case(
        n_target, 0, n_group, int, SINGLE_EXPTIME, seed=(n_target, 0, n_group, 103)
    )
    fake_logger = _FakeLogger()

    sample_out, fh, fh_pct, n, n_pct, sub_l = completion_rates(
        sample, point_l, SINGLE_EXPTIME, logger=fake_logger
    )

    n_sub = len(sub_l)
    zero_row = [[0] * (n_sub + 1)]

    assign = np.asarray(sample_out["exptime_assign"])
    assert np.array_equal(assign, np.zeros(n_target, dtype=assign.dtype))
    assert np.issubdtype(assign.dtype, np.integer)

    arrays = [fh, fh_pct, n, n_pct]
    for arr in arrays:
        assert np.array_equal(arr, zero_row)
    # Four distinct objects, not the same array handed back four times.
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            assert arrays[i] is not arrays[j]

    assert fake_logger.messages == ["No PPC is determined. Return zeros."]


def test_zero_pointings_with_no_logger_does_not_raise():
    n_target, n_group = 6, 3
    sample, point_l = _make_case(
        n_target, 0, n_group, float, SINGLE_EXPTIME, seed=(n_target, 0, n_group, 104)
    )

    # logger defaults to None; must not raise, and nothing is there to log to.
    sample_out, _, _, _, _, _ = completion_rates(
        sample, point_l, SINGLE_EXPTIME, logger=None
    )

    assign = np.asarray(sample_out["exptime_assign"])
    assert np.array_equal(assign, np.zeros(n_target, dtype=assign.dtype))


def test_exptime_assign_equals_min_exptime_and_single_exptime_times_hits():
    """`assign = min(exptime, single_exptime * hits)`, with `hits` counted
    independently here directly from the raw allocation lists: unique codes
    per pointing, unknown codes ignored. This is the order-independence fact
    the delta update in `_completion_curve` relies on."""
    n_target, n_point, n_group = 60, 20, 4
    sample, point_l = _make_case(
        n_target,
        n_point,
        n_group,
        float,
        SINGLE_EXPTIME,
        seed=(n_target, n_point, n_group, 105),
    )
    codes = np.asarray(sample["ob_code"])
    code_to_idx = {c: i for i, c in enumerate(codes)}

    hits = np.zeros(n_target, dtype=np.int64)
    for row in point_l["allocated_targets"]:
        unique_known = {c for c in row if c in code_to_idx}
        for c in unique_known:
            hits[code_to_idx[c]] += 1

    result = completion_rates(sample.copy(), point_l, SINGLE_EXPTIME)
    assign = np.asarray(result[0]["exptime_assign"])
    expected = np.minimum(
        np.asarray(sample["exptime"], dtype=float), SINGLE_EXPTIME * hits
    )
    assert np.allclose(assign, expected, rtol=0, atol=1e-9)


def test_final_exptime_assign_is_independent_of_pointing_order():
    """Shuffling the pointings (and re-randomizing ppc_priority, so the sort
    order actually changes) must not change the final `exptime_assign` --
    only the cumulative curves depend on order."""
    n_target, n_point, n_group = 60, 20, 3
    sample, point_l = _make_case(
        n_target,
        n_point,
        n_group,
        float,
        SINGLE_EXPTIME,
        seed=(n_target, n_point, n_group, 106),
    )
    rng = np.random.default_rng(107)

    result_a = completion_rates(sample.copy(), point_l, SINGLE_EXPTIME)

    shuffled_order = rng.permutation(len(point_l))
    point_l_shuffled = point_l[shuffled_order]
    point_l_shuffled["ppc_priority"] = rng.permutation(
        np.asarray(point_l_shuffled["ppc_priority"])
    )

    result_b = completion_rates(sample.copy(), point_l_shuffled, SINGLE_EXPTIME)

    assert np.array_equal(
        np.asarray(result_a[0]["exptime_assign"]),
        np.asarray(result_b[0]["exptime_assign"]),
    )


def test_curves_are_non_decreasing_along_pointings():
    n_target, n_point, n_group = 80, 25, 4
    sample, point_l = _make_case(
        n_target,
        n_point,
        n_group,
        float,
        SINGLE_EXPTIME,
        seed=(n_target, n_point, n_group, 110),
    )

    _, fh, fh_pct, n, n_pct, _ = completion_rates(
        sample.copy(), point_l, SINGLE_EXPTIME
    )

    assert np.all(np.diff(fh, axis=0) >= -1e-9)
    assert np.all(np.diff(fh_pct, axis=0) >= -1e-9)
    assert np.all(np.diff(n, axis=0) >= 0)
    assert np.all(np.diff(n_pct, axis=0) >= -1e-9)


def test_percentages_never_exceed_100():
    n_target, n_point, n_group = 80, 25, 4
    sample, point_l = _make_case(
        n_target,
        n_point,
        n_group,
        float,
        SINGLE_EXPTIME,
        seed=(n_target, n_point, n_group, 111),
    )

    _, _, fh_pct, _, n_pct, _ = completion_rates(sample.copy(), point_l, SINGLE_EXPTIME)

    assert np.all(fh_pct <= 100 + 1e-9)
    assert np.all(n_pct <= 100 + 1e-9)


def test_last_curve_row_matches_final_exptime_assign_sums():
    n_target, n_point, n_group = 100, 30, 5
    sample, point_l = _make_case(
        n_target,
        n_point,
        n_group,
        float,
        SINGLE_EXPTIME,
        seed=(n_target, n_point, n_group, 112),
    )

    sample_out, fh, _, _, _, sub_l = completion_rates(
        sample.copy(), point_l, SINGLE_EXPTIME
    )

    exptime_assign = np.asarray(sample_out["exptime_assign"], dtype=float)
    priority = np.asarray(sample_out["priority"])
    expected_last_row = [exptime_assign.sum() / 3600.0] + [
        exptime_assign[priority == g].sum() / 3600.0 for g in sub_l
    ]
    assert np.allclose(fh[-1], expected_last_row, rtol=0, atol=1e-9)


def test_unknown_and_duplicate_codes_do_not_change_the_result():
    """Codes not in `sample["ob_code"]`, and codes repeated within a single
    pointing, must not change anything compared to allocation lists cleaned
    of both -- `np.isin` and per-pointing `np.unique` already absorb them."""
    n_target, n_point, n_group = 40, 15, 3
    sample, point_l = _make_case(
        n_target,
        n_point,
        n_group,
        int,
        SINGLE_EXPTIME,
        seed=(n_target, n_point, n_group, 113),
    )
    known_codes = set(np.asarray(sample["ob_code"]))

    cleaned = [
        list(dict.fromkeys(c for c in row if c in known_codes))
        for row in point_l["allocated_targets"]
    ]
    point_l_clean = point_l.copy()
    point_l_clean["allocated_targets"] = Column(cleaned, dtype=object)

    dirty = completion_rates(sample.copy(), point_l, SINGLE_EXPTIME)
    clean = completion_rates(sample.copy(), point_l_clean, SINGLE_EXPTIME)

    assert np.array_equal(
        np.asarray(dirty[0]["exptime_assign"]), np.asarray(clean[0]["exptime_assign"])
    )
    for idx in (1, 2, 3, 4):
        assert np.allclose(dirty[idx], clean[idx], rtol=0, atol=1e-9)
    assert dirty[5] == clean[5]


def test_float_priorities_like_1_0_and_2_0_group_correctly():
    """Hand-built, generator-independent check: repeated float priority
    values must share a group, and distinct float values must not."""
    sample = Table(
        {
            "ob_code": np.array(["a", "b", "c", "d"]),
            "priority": np.array([1.0, 1.0, 2.0, 2.0]),
            "exptime": np.array([900.0, 900.0, 900.0, 900.0]),
        }
    )
    point_l = Table()
    point_l["ppc_priority"] = Column([0.0], dtype=np.float64)
    point_l["allocated_targets"] = Column([["a", "c"]], dtype=object)

    _, fh, _, n, _, sub_l = completion_rates(sample.copy(), point_l, SINGLE_EXPTIME)

    assert sub_l == [1.0, 2.0]
    # column 0 = whole sample, column 1 = priority 1.0, column 2 = priority 2.0
    assert n[0].tolist() == [2, 1, 1]
    assert np.allclose(fh[0], [1800.0 / 3600.0, 900.0 / 3600.0, 900.0 / 3600.0])
