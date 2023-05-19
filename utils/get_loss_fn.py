import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class irm_loss(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    @autocast()
    def forward(self, outputs, labels, split_all, idx):
        if self.num_classes == 1:
            loss_value = F.binary_cross_entropy_with_logits(outputs.squeeze(1), labels.float(), reduction='none')
            scale = torch.ones((1, outputs.size(-1))).cuda().requires_grad_()
            penalty = F.binary_cross_entropy_with_logits((outputs * scale).squeeze(1), labels.float(), reduction='none')
        else:
            loss_value = F.cross_entropy(outputs, labels, reduction='none')
            scale = torch.ones((1, outputs.size(-1))).cuda().requires_grad_()
            penalty = F.cross_entropy(outputs * scale, labels, reduction='none')
        split = split_all[idx]
        penalty = (split * penalty.unsqueeze(-1) / penalty.shape[0]).sum(0)
        erm_risk = (split * loss_value.unsqueeze(-1) / loss_value.shape[0]).sum(0)
        irm_risk_list = []
        for index in range(penalty.size(0)):
            irm_risk = torch.autograd.grad(penalty[index], [scale], create_graph=True)[0]
            irm_risk_list.append(torch.sum(irm_risk ** 2))
        erm = erm_risk.sum()
        penalty = torch.stack(irm_risk_list).sum()
        irm_risk_final = erm + 0 * penalty
        # scale_multi = loss_scale(irm_risk_final, 50)
        # irm_risk_final *= scale_multi
        return irm_risk_final, erm, penalty

class em_loss(torch.nn.Module):
    def __init__(self, n_envs):
        super().__init__()
        self.n_envs = n_envs
    
    def forward(self, x, mu):
        dims = x.shape[1]
        eps = torch.finfo(torch.float32).eps
        eps_cov = (torch.eye(dims) * torch.finfo(torch.float16).eps).cuda()
        # mu = torch.randn(self.n_envs, dims).cuda()
        cov = torch.stack(self.n_envs * [torch.eye(dims)]).cuda()
        pi = torch.ones(self.n_envs).div_(self.n_envs).cuda()
        x = x.unsqueeze(1)
        converged = False
        i = 0
        while not converged:
            prev_mu = mu.clone()
            prev_cov = cov.clone()
            prev_pi = pi.clone()
            print('start')
            h = MultivariateNormal(mu, cov)
            print('middle')
            llhood = h.log_prob(x)
            print('end')
            weighted_llhood = llhood + torch.log(pi)
            log_sum_lhood = torch.logsumexp(weighted_llhood, dim=1, keepdim=True)
            log_posterior = weighted_llhood - log_sum_lhood
            posterior = torch.exp(log_posterior.unsqueeze(2))
            pi = torch.sum(posterior, dim=0)
            if torch.any(pi == 0):
                pi = pi + eps
            mu = 1 * (torch.sum(posterior * x, dim=0) / pi) + 0 * prev_mu
            cov = 1 * (torch.matmul((posterior * (x - mu)).permute(1, 2, 0), (x - mu).permute(1, 0, 2)) / pi.unsqueeze(2) + eps_cov) + 0 * prev_cov
            pi = pi.squeeze() / x.shape[0]
            allclose = torch.allclose(mu, prev_mu) and torch.allclose(cov, prev_cov) and torch.allclose(pi, prev_pi)
            i += 1
            max_iter = i > 0
            converged = allclose or max_iter
        scale_multi = loss_scale(log_sum_lhood.mean(), -50)
        loss = log_sum_lhood.mean() * scale_multi
        split_all = F.one_hot(weighted_llhood.detach().argmax(dim=1))
        # return loss, log_sum_lhood.mean().item(), split_all
        return -log_sum_lhood.mean(), log_sum_lhood.mean().item(), split_all, mu.detach()
        
def loss_scale(loss, default_scale=100):
    with torch.no_grad():
        scale =  default_scale / torch.abs(loss.clone().detach())
    return scale