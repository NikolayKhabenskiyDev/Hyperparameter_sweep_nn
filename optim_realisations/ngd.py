import copy
import torch
import warnings

def Armijo(func, x, g, d, lr, rho, c1, iter):

    for _ in range(iter):
        f_prev = float(func())
        g_prev = g
        gtd_prev = torch.dot(g_prev.reshape(-1), d.reshape(-1))

        x.data = x.data + lr * d
        f_new = float(func())

        if not f_new <= float(f_prev + c1 * lr * gtd_prev):
            alpha = lr
            lr = lr * rho
        else:
            alpha = lr
            x.data = x.data - alpha * d
            break

        x.data = x.data - alpha * d

    return alpha

class NGD(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        eps = 1e-3,
        method = 'PRP',
        line_search = 'Armijo',
        c1 = 1e-4,
        c2 = 0.1,
        c = 0.35,
        lr = 1,
        rho = 0.5,
        max_ls = 10,
    ):
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))

        if method not in ['FR', 'PRP',]:
            raise ValueError('Invalid method: {}'.format(method))

        if line_search not in ['Armijo', 'Strong_Wolfe', 'None',]:
            raise ValueError('Invalid line search: {}'.format(line_search))
        elif line_search == 'None':
            warnings.warn("Unless loss is a quadratic function, this is not correct")

        if not (0.0 < c1 < 1.0):
            raise ValueError('Invalid c1 value: {}'.format(c1))

        if not (c1 < c2 < 1.0):
            raise ValueError('Invalid c2 value: {}'.format(c2))

        if not (0.0 < c < 0.5):
            raise ValueError('Invalid c value: {}'.format(c))

        if lr < 0.0:
            raise ValueError('Invalid lr value: {}'.format(lr))

        if not (0.0 < rho < 1.0):
            raise ValueError('Invalid rho value: {}'.format(rho))

        if max_ls % 1 != 0 or max_ls <= 0:
            raise ValueError('Invalid max_ls value: {}'.format(max_ls))

        defaults = dict(
            eps=eps,
            method = method,
            line_search=line_search,
            c1 = c1,
            c2 = c2,
            c = c,
            lr = lr,
            rho = rho,
            max_ls = max_ls,
        )

        super(NGD, self).__init__(params, defaults)

    def _get_A(p, d_p):

        A = torch.stack(
                        [torch.autograd.grad(
                            d_p[i],
                            p,
                            grad_outputs=torch.ones_like(d_p[i]),
                            retain_graph=True)[0]
                        for i in range(0, len(d_p))])

        return A

    def Exact(A, d_p, d):
        rdotr = torch.dot(-d, d_p.data)

        z = torch.matmul(A, d)

        alpha = rdotr / torch.matmul(d, z)

        return alpha

    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                state = self.state[p]

                method = group['method']
                line_search = group['line_search']
                c1 = group['c1']
                c2 = group['c2']
                c = group['c']
                lr = group['lr']
                rho = group['rho']
                eps = group['eps']
                max_ls = group['max_ls']

                if len(state) == 0:
                    state['g'] = copy.deepcopy(d_p.data)

                    if torch.norm(state['g']) < group['eps']:
                        return loss

                    state['d'] = copy.deepcopy(-d_p.data)

                    state['index'] = True

                    state['step'] = 0

                    state['alpha'] = lr
                else:
                    if method == 'FR':
                        state['beta'] = torch.norm(d_p.data) ** 2 / torch.norm(state['g']) ** 2

                    elif method == 'PRP':
                        state['beta'] = torch.dot(
                            d_p.data.reshape(-1),
                            (d_p.data.reshape(-1) - state['g'].reshape(-1))) / torch.norm(state['g']) ** 2

                    state['g'] = copy.deepcopy(d_p.data)

                    if torch.norm(state['g']) < group['eps']:
                        return loss

                    state['d'] = -state['g'] + state['beta'] * state['d']

                    state['index'] = False

                if line_search == 'Armijo':
                    state['alpha'] = Armijo(closure, p, state['g'], state['d'], state['alpha'], rho, c1, max_ls)

                p.data.add_(state['d'], alpha=state['alpha'])

        return loss
