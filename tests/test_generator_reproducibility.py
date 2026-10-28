import importlib
import numpy as np
import pandas as pd
import generate_data


def test_default_generation_is_reproducible_across_calls():
    first = generate_data.generate_customers(20)
    generate_data.generate_customers(7)
    pd.testing.assert_frame_equal(first, generate_data.generate_customers(20))


def test_explicit_seed_controls_all_draws():
    first = generate_data.generate_customers(30, seed=123)
    pd.testing.assert_frame_equal(first, generate_data.generate_customers(30, seed=123))
    assert not first.equals(generate_data.generate_customers(30, seed=456))


def test_import_does_not_reset_global_numpy_random_state():
    original = np.random.get_state()
    try:
        np.random.seed(12345)
        before = np.random.get_state()
        importlib.reload(generate_data)
        after = np.random.get_state()
    finally:
        np.random.set_state(original)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
