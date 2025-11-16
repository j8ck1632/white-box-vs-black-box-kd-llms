import numpy as np

from offline_teacher_data import _to_py, _to_numpy_array


def test_to_py_handles_nested_object_arrays():
    head0 = np.array(
        [
            np.array([0.1, 0.2], dtype=np.float16),
            np.array([0.3, 0.4], dtype=np.float16),
        ],
        dtype=object,
    )
    head1 = np.array(
        [
            np.array([0.5, 0.6], dtype=np.float16),
            np.array([0.7, 0.8], dtype=np.float16),
        ],
        dtype=object,
    )
    attention = np.array([head0, head1], dtype=object)

    python_value = _to_py(attention)

    assert isinstance(python_value, list)
    assert python_value[0][0] == [np.float16(0.1), np.float16(0.2)]
    assert python_value[1][1] == [np.float16(0.7), np.float16(0.8)]


def test_to_numpy_array_round_trip_after_to_py():
    head = [
        [0.0, 1.0, 0.0],
        [0.2, 0.3, 0.5],
        [0.4, 0.4, 0.2],
    ]
    nested_list = [head, head]

    arr = _to_numpy_array(nested_list, np.float16)

    assert arr.shape == (2, 3, 3)
    assert np.allclose(arr[0, 1], np.array([0.2, 0.3, 0.5], dtype=np.float16))

