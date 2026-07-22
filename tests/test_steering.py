import pytest

torch = pytest.importorskip("torch")

from cross_interface_steering.steering import _last_token_indices


def test_last_token_indices_support_left_and_right_padding() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    assert _last_token_indices(attention_mask).tolist() == [2, 4, 4]
