from scipy.optimize import line_search
import numpy as np
import torch

class SteepestGD(torch.optim.Optimizer):
    def __init__(self, params, alternate_lr=1e-3):
        defaults = dict(alternate_lr=alternate_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def get_flatten_numpy_params(self):
        w_list = []
        grad_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                w_list.append(p.data.ravel().detach().clone().numpy())
                grad_list.append(p.grad.ravel().detach().clone().numpy())
        w = np.concatenate(w_list)
        grad = np.concatenate(grad_list)
        return w, grad

    @torch.no_grad()
    def set_params_from_flatten_numpy(self, params):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                size = p.numel()
                update = (
                    torch.tensor(params[idx : idx + size].reshape(p.shape))
                    .requires_grad_(True)
                    .to(p.data)
                )
                p.data.copy_(update)
                idx += size

    def f(self, w):
        self.set_params_from_flatten_numpy(w)
        f = self._closure().item()
        return f

    def grad_f(self, w):
        self.set_params_from_flatten_numpy(w)
        self._closure()
        w, grad = self.get_flatten_numpy_params()
        return grad

    def step(self, closure):
        assert closure is not None
        self._closure = closure

        f = closure()
        w, grad = self.get_flatten_numpy_params()
        p = -grad
        alpha, _, _, _, _, _ = line_search(self.f, self.grad_f, w, p, grad)

        if alpha is None:
            w_new = w + self.defaults["alternate_lr"] * p
        else:
            w_new = w + alpha * p

        self.set_params_from_flatten_numpy(w_new)
        return f
