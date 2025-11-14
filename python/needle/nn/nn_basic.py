from typing import Any

import needle.init as init
from needle.autograd import Tensor
from needle.backend.device import Device


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Parameter]:
    """Recursively collect all Parameter instances contained in a nested structure.

    Supports arbitrary nesting of `Module`, `dict`, `list`, and `tuple` objects.

    Args:
        value: An object that may contain parameters.

    Returns:
        A flat list of `Parameter` objects discovered within `value`.
    """
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for _, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    """Recursively collect all child Module instances from a nested structure.

    Args:
        value: An object that may contain `Module` instances.

    Returns:
        A flat list of `Module` objects discovered within `value`.
    """
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for _, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    """Base class for neural network modules.

    Provides:
    - Parameter discovery via `parameters()`
    - Child module discovery via `_children()`
    - Training/evaluation mode toggling via `train()` and `eval()`
    - A callable interface that forwards to `forward(...)`
    """

    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Parameter]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        """Return a flat list of all submodules contained in this module."""
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        """Switch the module and all children to evaluation mode."""
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        """Switch the module and all children to training mode."""
        self.training = True
        for m in self._children():
            m.training = True

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Compute the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Make modules callable: delegates to `forward`."""
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Return the input unchanged."""
        (x,) = args
        return x


class Linear(Module):
    """Applies a linear transformation: y = x @ W (+ b).

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias to the output.
        device: Device on which to allocate parameters.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Device | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )

        self.bias = (
            Parameter(
                init.kaiming_uniform(
                    out_features, 1, device=device, dtype=dtype
                ).transpose()
            )
            if bias
            else None
        )

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Compute x @ weight (+ bias), where:
        - x has shape (batch, in_features)
        - weight has shape (in_features, out_features)
        - bias (if present) has shape (1, out_features)
        """
        (X,) = args
        out = X @ self.weight
        if self.bias:
            out += self.bias.broadcast_to(out.shape)
        return out


class Flatten(Module):
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Flatten all dimensions except the batch dimension."""
        (X,) = args
        flattened_dim = 1
        for dim in X.shape[1:]:
            flattened_dim *= dim
        return X.reshape((X.shape[0], flattened_dim))


class ReLU(Module):
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Apply the rectified linear unit activation elementwise."""
        (x,) = args
        return x.relu()


class Sequential(Module):
    """A container that applies a sequence of modules in order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Apply each contained module to the input in sequence."""
        (x,) = args
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    """Cross-entropy loss between logits and integer class labels using log-sum-exp."""

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Compute mean cross-entropy loss.

        Args:
            logits: Unnormalized scores of shape (batch, num_classes).
            y: Integer class labels of shape (batch,).

        Returns:
            A scalar tensor with the mean loss over the batch.
        """
        (logits, y) = args
        batch_size, num_classes = logits.shape

        # Create a one-hot encoding of the labels
        one_hot_y = init.one_hot(num_classes, y)
        log_sum_exp = logits.logsumexp(axes=(1,))

        # We sum over axes 1 to get the logit of the correct class for each example.
        logit_y = (logits * one_hot_y).sum(axes=(1,))
        loss_per_example = log_sum_exp - logit_y

        return loss_per_example.sum() / batch_size


def normalize(x: Tensor, mean: Tensor, var: Tensor, eps: float) -> Tensor:
    """Normalize tensor x given mean and variance with numerical stability epsilon."""
    return (x - mean.broadcast_to(x.shape)) / (var.broadcast_to(x.shape) + eps) ** 0.5


def rescale_and_shift(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply affine transformation: weight * x + bias (with broadcasting to x.shape)."""
    return weight.broadcast_to(x.shape) * x + bias.broadcast_to(x.shape)


class BatchNorm1d(Module):
    """Batch Normalization over a mini-batch of 1D activations.

    Maintains running estimates of mean and variance for evaluation.

    Args:
        dim: Number of features in the input.
        eps: Small constant for numerical stability.
        momentum: Momentum for running statistics update.
        device: Device on which to allocate parameters.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Device | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

        # Running mean and variance are used at test time to normalize the input.
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Normalize input across the batch, then apply learnable scale and bias.

        Uses running statistics when in evaluation mode.
        """
        (x,) = args
        if not self.training:
            # At test time, we use the running mean and variance to normalize the input.
            normalized_x = normalize(x, self.running_mean, self.running_var, self.eps)
            return rescale_and_shift(normalized_x, self.weight, self.bias)

        batch_size = x.shape[0]
        # Compute the mean and variance of the batch.
        mean = x.sum(axes=0) / batch_size
        diff = x - mean.broadcast_to(x.shape)
        var = (diff**2).sum(axes=0) / batch_size

        # Update the running mean and variance to be used at test time.
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        # Normalize the input and apply the weight and bias.
        normalized_x = normalize(x, mean, var, self.eps)
        return rescale_and_shift(normalized_x, self.weight, self.bias)


class LayerNorm1d(Module):
    """Layer Normalization over the last dimension of 1D activations.

    Args:
        dim: Number of features in the input.
        eps: Small constant for numerical stability.
        device: Device on which to allocate parameters.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Device | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

        # The weight of a feature is the same for each example in a mini-batch.
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        # The bias of a feature is the same for each example in a mini-batch.
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Normalize each example across its feature dimension, then scale and shift."""
        (x,) = args

        mean = x.sum(axes=1, keepdims=True) / self.dim
        diff = x - mean.broadcast_to(x.shape)
        var = (diff**2).sum(axes=1, keepdims=True) / self.dim

        normalized_x = normalize(x, mean, var, self.eps)
        return rescale_and_shift(normalized_x, self.weight, self.bias)


class Dropout(Module):
    """Randomly zero a fraction p of elements during training for regularization."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Apply dropout mask during training; pass inputs through during evaluation."""
        (x,) = args
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)
            return x * mask / (1 - self.p)
        else:
            return x


class Residual(Module):
    """Wrap a module with a residual (skip) connection: f(x) + x."""

    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """Compute f(x) + x for the wrapped function f."""
        (x,) = args
        return self.fn(x) + x
