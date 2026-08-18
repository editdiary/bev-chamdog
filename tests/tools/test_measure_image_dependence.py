"""이미지 의존도 진단의 순열 계약.

이 도구의 결론("이미지를 무시하고 있다")은 순열이 실제로 **다른 장면**을 물어다 주는지에
전적으로 달려 있다. 인접 프레임끼리 섞으면 장면이 거의 같아 "예측이 안 변했다"가 아무것도
증명하지 못한다.
"""
import numpy as np
import pytest

from tools.measure_image_dependence import shuffled_order


def test_no_sample_keeps_its_own_images():
    """자기 이미지를 그대로 받는 샘플이 있으면 그 샘플은 진단에 기여하지 않는다."""
    for count in (2, 3, 10, 37, 75, 100):
        order = shuffled_order(count)
        assert not (order == np.arange(count)).any(), count


def test_the_order_is_a_permutation():
    """순열이 아니면 어떤 이미지는 두 번 쓰이고 어떤 이미지는 안 쓰인다 -- 비교가 편향된다."""
    for count in (2, 5, 75):
        assert sorted(shuffled_order(count).tolist()) == list(range(count))


def test_samples_are_paired_with_distant_frames_not_neighbours():
    """이웃 프레임은 장면이 거의 같다. 최소한 데이터셋 절반만큼 떨어져야 한다.

    `+1`처럼 가까운 오프셋으로 바뀌면 이 테스트가 깨진다 -- 그 순간 도구의 결론이 무의미해진다.
    """
    count = 75
    order = shuffled_order(count)
    distance = np.minimum(np.abs(order - np.arange(count)), count - np.abs(order - np.arange(count)))

    assert distance.min() >= count // 2 - 1, distance.min()


def test_a_single_sample_cannot_be_shuffled():
    """샘플이 하나면 섞을 수 없다. 조용히 자기 이미지를 돌려주면 진단이 거짓 음성을 낸다."""
    with pytest.raises(ValueError):
        shuffled_order(1)


def test_the_order_is_deterministic():
    """실행마다 숫자가 흔들리면 런 간 비교가 안 된다."""
    assert shuffled_order(75).tolist() == shuffled_order(75).tolist()
