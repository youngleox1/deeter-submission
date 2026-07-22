from src.lr_schedule import cosine_warmup_multiplier


def test_warmup_ramps_linearly_from_near_zero_to_one():
    total_steps, warmup_steps = 1000, 100
    m_first = cosine_warmup_multiplier(1, total_steps, warmup_steps)
    m_mid_warmup = cosine_warmup_multiplier(50, total_steps, warmup_steps)
    m_end_warmup = cosine_warmup_multiplier(warmup_steps, total_steps, warmup_steps)

    assert abs(m_first - 1 / warmup_steps) < 1e-9
    assert abs(m_mid_warmup - 50 / warmup_steps) < 1e-9
    assert m_first < m_mid_warmup < m_end_warmup


def test_cosine_decay_reaches_min_lr_ratio_at_final_step():
    total_steps, warmup_steps, min_lr_ratio = 1000, 100, 0.1
    m_final = cosine_warmup_multiplier(total_steps, total_steps, warmup_steps, min_lr_ratio)
    assert abs(m_final - min_lr_ratio) < 1e-6


def test_cosine_decay_peaks_at_one_right_after_warmup():
    total_steps, warmup_steps = 1000, 100
    m_at_warmup_end = cosine_warmup_multiplier(warmup_steps, total_steps, warmup_steps)
    assert abs(m_at_warmup_end - 1.0) < 1e-6


def test_multiplier_is_monotonically_decreasing_after_warmup():
    total_steps, warmup_steps = 1000, 100
    steps = [warmup_steps, 300, 500, 700, 900, total_steps]
    multipliers = [cosine_warmup_multiplier(s, total_steps, warmup_steps) for s in steps]
    assert multipliers == sorted(multipliers, reverse=True)


def test_zero_warmup_steps_starts_at_full_multiplier():
    # step=1 of 1000 with no warmup is already fractionally into the
    # cosine decay (progress=1/1000), so it's very close to but not
    # bit-exact 1.0 -- tolerance reflects that, not a bug.
    total_steps = 1000
    m = cosine_warmup_multiplier(1, total_steps, warmup_steps=0)
    assert abs(m - 1.0) < 1e-4


def test_warmup_longer_than_run_stays_in_linear_ramp_throughout():
    # warmup_steps > total_steps: every reachable step is still < warmup_steps,
    # so the run never leaves the linear ramp phase (the decay branch is
    # unreachable here, not merely untested).
    m_mid = cosine_warmup_multiplier(50, total_steps=100, warmup_steps=200)
    assert abs(m_mid - 50 / 200) < 1e-9
    m_at_end = cosine_warmup_multiplier(100, total_steps=100, warmup_steps=200)
    assert abs(m_at_end - 100 / 200) < 1e-9


def test_warmup_exactly_covering_entire_run_reaches_one_at_final_step():
    # warmup_steps == total_steps: the final step (step == warmup_steps,
    # not < it) falls through to the "no decay phase" branch -> 1.0.
    m_at_end = cosine_warmup_multiplier(100, total_steps=100, warmup_steps=100)
    assert abs(m_at_end - 1.0) < 1e-9
