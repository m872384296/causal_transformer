import torch
import torch.nn.functional as F

class irm_loss(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, outputs, labels, soft_split_all, idx):
        if self.num_classes == 1:
            loss_value = F.binary_cross_entropy_with_logits(outputs.squeeze(1), labels.float(), reduction='none')
            scale = torch.ones((1, outputs.size(-1))).cuda().requires_grad_()
            penalty = F.binary_cross_entropy_with_logits((outputs * scale).squeeze(1), labels.float(), reduction='none')
        else:
            loss_value = F.cross_entropy(outputs, labels, reduction='none')
            scale = torch.ones((1, outputs.size(-1))).cuda().requires_grad_()
            penalty = F.cross_entropy(outputs * scale, labels, reduction='none')
        split_logits = F.log_softmax(soft_split_all, dim=-1)
        hard_split_all = F.gumbel_softmax(split_logits, tau=1, hard=True)
        hard_split = hard_split_all[idx]
        penalty = (hard_split * penalty.unsqueeze(-1) / (hard_split.sum(0) + 1e-20)).sum(0)
        erm_risk = (hard_split * loss_value.unsqueeze(-1) / (hard_split.sum(0) + 1e-20)).sum(0)
        irm_risk_list = []
        for index in range(penalty.size(0)):
            irm_risk = torch.autograd.grad(penalty[index], [scale], create_graph=True)[0]
            irm_risk_list.append(torch.sum(irm_risk ** 2))
        erm = erm_risk.mean()
        penalty = torch.stack(irm_risk_list).mean()
        irm_risk_final = erm + 1e6 * penalty
        scale_multi = irm_scale(irm_risk_final, 50)
        irm_risk_final *= scale_multi
        # return irm_risk_final, erm, penalty
        return erm
    
def irm_scale(irm_loss, default_scale=100):
    with torch.no_grad():
        scale =  default_scale / irm_loss.clone().detach()
    return scale