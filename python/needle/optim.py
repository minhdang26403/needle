from collections import defaultdict

from needle.autograd import Tensor
from needle.nn import Parameter


class Optimizer:
    def __init__(self, params: list[Parameter]):
        self.params = params

    def step(self) -> None:
        raise NotImplementedError()

    def reset_grad(self) -> None:
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u: defaultdict[int, Tensor] = defaultdict()
        self.weight_decay = weight_decay

    def step(self) -> None:
        for i, p in enumerate(self.params):
            # Apply weight decay to the gradient.
            # Note: This operation should not be tracked for gradient computation
            # so operate on the underlying data arrays.
            assert p.grad is not None
            grad = p.grad.data + self.weight_decay * p.data

            # Get the old velocity.
            v_old = self.u[i]

            # Compute the new velocity using the momentum rule
            v_new = self.momentum * v_old + (1 - self.momentum) * grad

            # Update the parameter using the new velocity.
            # This must be done in a way that doesn't build up the computation graph,
            # so we operate on the raw data arrays.
            p.data = p.data - self.lr * v_new.data

            # Save the new velocity for the next iteration.
            self.u[i] = v_new

    def clip_grad_norm(self, max_norm: float = 0.25) -> None:
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m: defaultdict[int, Tensor] = defaultdict()
        self.v: defaultdict[int, Tensor] = defaultdict()

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            assert p.grad is not None
            grad = p.grad.data + self.weight_decay * p.data

            # Update the first moment and second moment estimates.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # Compute the bias-corrected first and second moment estimates.
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update the parameter using the Adam update rule.
            p.data = p.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
