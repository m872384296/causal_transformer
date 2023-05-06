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
        irm_risk_final = erm + 1e0 * penalty
        # scale_multi = loss_scale(irm_risk_final, 50)
        # irm_risk_final *= scale_multi
        return irm_risk_final, erm, penalty
    
# class unshuffle(torch.nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
    
#     @autocast()
#     def forward(self, epoch, net, train_loader, args, training_opt, variance_opt, loss_function, optimizer, warmup_scheduler):
#         env_num = variance_opt['n_env']
#         assert isinstance(net, list)
#         for net_ in net:
#             net_.train()
#         train_correct = 0.
#         train_image_num = 0

#         if variance_opt['erm_flag']:
#             erm_dataloader_iterator = iter(train_loader[1])
#         for batch_index, data_env in enumerate(train_loader[0]):
#             if variance_opt['erm_flag']:
#                 try:
#                     data_env_erm = next(erm_dataloader_iterator)
#                 except StopIteration:
#                     erm_dataloader_iterator = iter(train_loader[1])
#                     data_env_erm = next(erm_dataloader_iterator)


#             if epoch <= training_opt['warm']:
#                 for warmup_scheduler_ in warmup_scheduler:
#                     warmup_scheduler_.step()

#             env_dic_nll = []
#             env_dic_nll_spurious = []
#             env_dic_acc = []
#             images_all = []
#             labels_all = []
#             Erm_loss = torch.Tensor([0.])
#             train_nll_spurious = torch.Tensor([0.])


#             for edx, env in enumerate(data_env):
#                 images, labels, env_idx = env
#                 assert env_idx[0] == edx
#                 if args.gpu:
#                     labels = labels.cuda()
#                     images = images.cuda()
#                 images_all.append(images)
#                 labels_all.append(labels)
#                 causal_feature, spurious_feature, mix_feature = net[-1](images)
#                 causal_outputs = net[edx](causal_feature)
#                 batch_correct, train_acc = cal_acc(causal_outputs, labels)
#                 train_correct += batch_correct
#                 train_image_num += labels.size(0)
#                 env_dic_nll.append(loss_function(causal_outputs, labels))
#                 env_dic_acc.append(train_acc)

#             train_nll = torch.stack(env_dic_nll).mean()
#             train_acc = torch.stack(env_dic_acc).mean()

#             ### 1. update feature extractor and classifier for irm
#             ### ERM Loss
#             loss = train_nll.clone()
#             ### Variance Loss
#             penalty_weight = float(variance_opt['penalty_weight']) if epoch >= variance_opt['penalty_anneal_iters'] else 1.0
#             # penalty_weight = 0.0

#             try:
#                 W_mean = torch.stack([net_.fc.weight for net_ in net[:env_num]], 0).mean(0)
#                 var_penalty = [(torch.norm(net_.fc.weight - W_mean, p=2) / torch.norm(net_.fc.weight, p=1)) ** 2 for net_ in net[:env_num]]
#             except:
#                 W_mean = torch.stack([net_.module.fc.weight for net_ in net[:env_num]], 0).mean(0)
#                 var_penalty = [(torch.norm(net_.module.fc.weight - W_mean, p=2) / torch.norm(net_.module.fc.weight, p=1))**2 for net_ in net[:env_num]]
#             loss_penalty = sum(var_penalty) / len(var_penalty)
#             loss += penalty_weight * loss_penalty

#             ### 2. update with erm loss
#             if variance_opt['erm_flag']:
#                 images_erm, labels_erm = data_env_erm
#                 if args.gpu:
#                     labels_erm = labels_erm.cuda()
#                     images_erm = images_erm.cuda()
#                 if 'mixup' in training_opt and training_opt['mixup'] == True:
#                     inputs_erm, targets_a_erm, targets_b_erm, lam = mixup_data(images_erm, labels_erm, use_cuda=True)
#                     images_erm, targets_a_erm, targets_b_erm = map(Variable, (inputs_erm, targets_a_erm, targets_b_erm))

#                 _, __, mix_feature_erm = net[-1](images_erm)
#                 mix_outputs = net[-2](mix_feature_erm)

#                 if 'mixup' in training_opt and training_opt['mixup'] == True:
#                     Erm_loss = mixup_criterion(loss_function, mix_outputs, targets_a_erm, targets_b_erm, lam)
#                 else:
#                     Erm_loss = loss_function(mix_outputs, labels_erm)

#                 loss += Erm_loss

#             for optimizer_ in optimizer:
#                 optimizer_.zero_grad()
#             loss.backward()
#             for optimizer_ in optimizer:
#                 optimizer_.step()

#             ### 3. update feature extractor with spurious feature
#             if variance_opt['sp_flag']:
#                 image_env = torch.cat(images_all, 0)
#                 _, spurious_feature_sp, __ = net[-1](image_env)
#                 # spurious_outputs = net[edx](spurious_feature_sp)
#                 try:
#                     W_mean = torch.stack([net_.fc.weight for net_ in net[:env_num]], 0).mean(0)
#                 except:
#                     W_mean = torch.stack([net_.module.fc.weight for net_ in net[:env_num]], 0).mean(0)
#                 spurious_outputs = torch.nn.functional.linear(spurious_feature_sp, W_mean)
#                 train_nll_spurious = smooth_loss(spurious_outputs, training_opt['classes'])
#                 optimizer[-1].zero_grad()
#                 train_nll_spurious.backward()
#                 optimizer[-1].step()


#             if batch_index % training_opt['print_batch'] == 0:
#                 print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tTrain_Loss: {:0.3f}\tNll_Loss: {:0.3f}\tPenalty: {:.2e}'
#                       '\tPenalty_W: {:.1f}\tErm_Loss: {:.3f}\tSp_Loss: {:.3f}\tLR: {:0.6f}\tAcc: {:0.4f}'.format(
#                     loss.item(),
#                     train_nll.item(),
#                     loss_penalty.item(),
#                     penalty_weight,
#                     Erm_loss.item(),
#                     train_nll_spurious.item(),
#                     optimizer[0].param_groups[0]['lr'],
#                     train_acc,
#                     epoch=epoch,
#                     trained_samples=batch_index * training_opt['batch_size'] + len(images),
#                     total_samples=len(train_loader[0].dataset)
#                 ))
#                 print('\n')

#         finish = time.time()
#         # train_acc_all = train_correct / sum([len(env_dataloader.dataset) for env_dataloader in train_loader])
#         train_acc_all = train_correct / train_image_num
#         print('epoch {} training time consumed: {:.2f}s \t Train Acc: {:.4f}'.format(epoch, finish - start, train_acc_all))

#         return train_acc_all

class em_loss(torch.nn.Module):
    def __init__(self, n_envs):
        super().__init__()
        self.n_envs = n_envs
    
    def forward(self, x):
        dims = x.shape[1]
        eps = torch.finfo(torch.float32).eps
        eps_cov = (torch.eye(dims) * torch.finfo(torch.float16).eps).cuda()
        mu = torch.randn(self.n_envs, dims).cuda()
        cov = torch.stack(self.n_envs * [torch.eye(dims)]).cuda()
        pi = torch.ones(self.n_envs).div_(self.n_envs).cuda()
        x = x.unsqueeze(1)
        converged = False
        i = 0
        while not converged:
            prev_mu = mu.clone()
            prev_cov = cov.clone()
            prev_pi = pi.clone()
            h = MultivariateNormal(mu, cov)
            llhood = h.log_prob(x)
            weighted_llhood = llhood + torch.log(pi)
            log_sum_lhood = torch.logsumexp(weighted_llhood, dim=1, keepdim=True)
            log_posterior = weighted_llhood - log_sum_lhood
            posterior = torch.exp(log_posterior.unsqueeze(2))
            pi = torch.sum(posterior, dim=0)
            if torch.any(pi == 0):
                pi = pi + eps
            mu = torch.sum(posterior * x, dim=0) / pi
            cov = torch.matmul((posterior * (x - mu)).permute(1, 2, 0), (x - mu).permute(1, 0, 2)) / pi.unsqueeze(2) + eps_cov
            # a, b, c, d = cov.split(1)
            # np.savetxt('a.csv', a.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('b.csv', b.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('c.csv', c.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('d.csv', d.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('post.csv', posterior.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('pi.csv', pi.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('x.csv', x.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            # np.savetxt('mu.csv', mu.squeeze().clone().detach().cpu().numpy(), delimiter=',')
            pi = pi.squeeze() / x.shape[0]
            allclose = torch.allclose(mu, prev_mu) and torch.allclose(cov, prev_cov) and torch.allclose(pi, prev_pi)
            i += 1
            # print(i)
            max_iter = i > 10
            converged = allclose or max_iter
        scale_multi = loss_scale(log_sum_lhood.mean(), -50)
        loss = log_sum_lhood.mean() * scale_multi
        split_all = F.one_hot(weighted_llhood.detach().argmax(dim=1))
        # return loss, log_sum_lhood.mean().item(), split_all
        return -log_sum_lhood.mean(), log_sum_lhood.mean().item(), split_all
        
def loss_scale(loss, default_scale=100):
    with torch.no_grad():
        scale =  default_scale / torch.abs(loss.clone().detach())
    return scale