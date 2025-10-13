import math
import torch
from torch.optim.optimizer import Optimizer
import torch.sparse as tsp


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("无效的学习率: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("无效的epsilon值: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("beta参数[0]需在[0,1)区间: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("beta参数[1]需在[0,1)区间: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def _sparse_addcmul(self, tensor, value, grad):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.zeros_like(grad) if grad.is_sparse else torch.tensor(tensor, dtype=grad.dtype, device=grad.device)
        if grad.is_sparse:
            sparse_grad = grad.coalesce()
            indices = sparse_grad.indices()
            values = sparse_grad.values()
            return tensor.index_add_(0, indices[0], values * value)
        # 修改此处：使用 add_ 和乘法替代 addcmul_
        return tensor.add_(grad * value)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse and not grad.is_coalesced():
                    grad = grad.coalesce()

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32, memory_format=torch.preserve_format)
                else:
                    state['exp_avg'] = state['exp_avg'].to(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].to(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1)
                self._sparse_addcmul(exp_avg, 1 - beta1, grad)

                exp_avg_sq.mul_(beta2)
                if grad.is_sparse:
                    grad_sq = grad.values().pow(2)
                    exp_avg_sq.index_add_(0, grad.indices()[0], (1 - beta2) * grad_sq)
                else:
                    exp_avg_sq.add_((1 - beta2) * grad.pow(2))

                buffered = group['buffer'][state['step'] % 10]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    beta2_t = beta2 ** state['step']
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                        ) / bias_correction1
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / bias_correction1
                    else:
                        step_size = -1
                    buffered[2] = step_size

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                if p.data.is_sparse:
                    p.data = p_data_fp32.to_sparse()
                else:
                    p.data.copy_(p_data_fp32)

        return loss