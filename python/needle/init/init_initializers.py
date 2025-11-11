import math
from typing import Any

from needle.autograd import Tensor
from needle.init import rand, randn


def xavier_uniform(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any
) -> Tensor:
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any
) -> Tensor:
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)


def kaiming_uniform(
    fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any
) -> Tensor:
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(
    fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any
) -> Tensor:
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std = gain * math.sqrt(1 / fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
