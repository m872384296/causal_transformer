from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score
from utils.build_net import weight_init

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def train_cls_module(config, rank, epoch, net, split_all, loss_fn, train_loader, optimizer, lr_scheduler, scaler, logger, writer):
    net.train()
    if rank == 0:
        iterator = tqdm(train_loader, desc=f'Training epoch {epoch}')
    else:
        iterator = train_loader
    num_steps = len(train_loader)
    label_all = torch.tensor([]).cuda(rank).long()
    idx_all = torch.tensor([]).long()
    y_all = torch.tensor([]).cuda(rank)
    conf_all = torch.tensor([])
    h_all = torch.tensor([])
    erm_all = 0
    penalty_all = 0
    for n_iter, (img, label, conf, idx) in enumerate(iterator):
        img = img.cuda(rank, non_blocking=True)
        label = label.cuda(rank, non_blocking=True)
        conf = conf.cuda(rank, non_blocking=True)
        idx = idx.cuda(rank, non_blocking=True)
        with autocast():
            y, h = net(img)
            loss, erm, penalty = loss_fn(y, label, split_all, idx)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dist.all_reduce(erm, op=dist.ReduceOp.SUM)
        dist.all_reduce(penalty, op=dist.ReduceOp.SUM)
        erm_reduce = erm.item() / config['world_size']
        penalty_reduce = penalty.item() / config['world_size']
        erm_all += erm.item() * label.shape[0]
        penalty_all += penalty.item() * label.shape[0]
        if rank == 0:
            writer.add_scalar('Learning rate 1', lr, epoch * num_steps + n_iter)
            writer.add_scalar('ERM', erm_reduce, epoch * num_steps + n_iter)
            writer.add_scalar('Penalty', penalty_reduce, epoch * num_steps + n_iter)
        conf_out = torch.zeros(config['world_size'] * conf.shape[0], conf.shape[1]).cuda(rank)
        dist.all_gather_into_tensor(conf_out, conf.float())
        conf_all = torch.cat((conf_all, conf_out.cpu()), 0)
        h_out = torch.zeros(config['world_size'] * h.shape[0], h.shape[1], h.shape[2]).cuda(rank)
        dist.all_gather_into_tensor(h_out, h)
        h_all = torch.cat((h_all, h_out.cpu()), 0)
        y_out = torch.zeros(config['world_size'] * y.shape[0], y.shape[1]).cuda(rank)
        dist.all_gather_into_tensor(y_out, y)
        y_all = torch.cat((y_all, y_out), 0)
        idx_out = torch.zeros(config['world_size'] * idx.shape[0]).cuda(rank).long()
        dist.all_gather_into_tensor(idx_out, idx)
        idx_all = torch.cat((idx_all, idx_out.cpu()), 0)
        label_out = torch.zeros(config['world_size'] * label.shape[0]).cuda(rank).long()
        print(label.tyoe())
        dist.all_gather_into_tensor(label_out, label)
        label_all = torch.cat((label_all, label_out), 0)
        if (n_iter + 1) % 1 == 0:
            if loss_fn.num_classes == 1:
                prob = torch.sigmoid(y_all).cpu().numpy()
                pred = np.where(prob > 0.5, 1, 0)
                f1 = f1_score(label_all.cpu().numpy(), pred)
                auc = roc_auc_score(label_all.cpu().numpy(), prob)
                if rank == 0:
                    writer.add_scalar('F1-score', f1, epoch * num_steps + n_iter)
                    writer.add_scalar('AUC', auc, epoch * num_steps + n_iter)
            else:
                pred = np.argmax(y_all.cpu().numpy(), axis=1)
                acc = (pred == label_all.cpu().numpy()).sum() / label_all.shape[0]
                if rank == 0:
                    writer.add_scalar('ACC', acc, epoch * num_steps + n_iter)
    erm_mean = erm_all / len(train_loader.dataset)
    penalty_mean = penalty_all / len(train_loader.dataset)
    if loss_fn.num_classes == 1:
        prob = torch.sigmoid(y_all).cpu().numpy()
        pred = np.where(prob > 0.5, 1, 0)
        f1 = f1_score(label_all.cpu().numpy(), pred)
        auc = roc_auc_score(label_all.cpu().numpy(), prob)
        logger.info(f'F1-score for training of epoch {epoch} is {f1:.3f}')
        logger.info(f'AUC for training of epoch {epoch} is {auc:.3f}')
    else:
        pred = np.argmax(y_all.cpu().numpy(), axis=1)
        acc = (pred == label_all.cpu().numpy()).sum() / label_all.shape[0]
        logger.info(f'Acc for training of epoch {epoch} is {acc:.3f}')
    logger.info(f'ERM of epoch {epoch} is {erm_mean:.3f}')
    logger.info(f'Penalty of epoch {epoch} is {penalty_mean:.3f}')
    return conf_all, h_all, idx_all

def train_spl_module(epoch, net, loss_fn, env_loader, optimizer, lr_scheduler, logger, writer):
    net.apply(weight_init)
    net.train()
    mu = torch.randn(loss_fn.n_envs, 128).cuda()
    for step in tqdm(range(10), desc=f'Spliting epoch {epoch}'):
        y_all = torch.tensor([]).cuda().half()
        for conf, h in env_loader:
            conf = conf.cuda(non_blocking=True)
            h = h.cuda(non_blocking=True)
            with autocast():
                y = net(conf, h)
            y_all = torch.cat((y_all, y.half()), 0)        
        loss, llhood, split_all, mu = loss_fn(y_all, mu)
        
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(y_all.cpu().detach().numpy())
        labels = split_all.cpu().detach().numpy()
        labels = labels.nonzero()[1]
        plt.scatter(result[:,0],result[:,1],s=1,c=labels)
        plt.savefig("./pic/" + str(epoch) + str(step) + ".png")
        plt.close()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning rate 2', lr, epoch * 10 + step)
        writer.add_scalar('Likelyhood', llhood, epoch * 10 + step)
    logger.info(f'Likelyhood is {llhood:.3f}')
    return split_all.cuda()
    
def validate(config, rank, epoch, net, val_loader, logger, writer):
    logger.info(f'Epoch {epoch} begin validating......')
    net.eval()
    if rank == 0:
        iterator = tqdm(val_loader, desc=f'Validating epoch {epoch}')
    else:
        iterator = val_loader
    num_steps = len(val_loader)
    with torch.no_grad():
        label_all = torch.tensor([]).cuda(rank)
        y_all = torch.tensor([]).cuda(rank)
        for n_iter, (img, label) in enumerate(iterator):
            img = img.cuda(rank, non_blocking=True)
            label = label.cuda(rank, non_blocking=True)
            with autocast():
                y, _ = net(img)
            y_out = torch.zeros(config['world_size'] * y.shape[0], y.shape[1]).cuda(rank)
            dist.all_gather_into_tensor(y_out, y.float())
            y_all = torch.cat((y_all, y_out), 0)
            label_out = torch.zeros(config['world_size'] * label.shape[0]).cuda(rank)
            dist.all_gather_into_tensor(label_out, label.float())
            label_all = torch.cat((label_all, label_out), 0)
            if (n_iter + 1) % 1 == 0:
                if net.module.num_classes == 1:
                    prob = torch.sigmoid(y_all).detach().cpu().numpy()
                    pred = np.where(prob > 0.5, 1, 0)
                    f1 = f1_score(label_all.detach().cpu().numpy(), pred)
                    auc = roc_auc_score(label_all.detach().cpu().numpy(), prob)
                    if rank == 0:
                        writer.add_scalar('val-F1-score', f1, epoch * num_steps + n_iter)
                        writer.add_scalar('val-AUC', auc, epoch * num_steps + n_iter)
                else:
                    pred = np.argmax(y_all.detach().cpu().numpy(), axis=1)
                    acc = (pred == label_all.detach().cpu().numpy()).sum() / label_all.shape[0]
                    if rank == 0:
                        writer.add_scalar('val-ACC', acc, epoch * num_steps + n_iter)
        if net.module.num_classes == 1:
            prob = torch.sigmoid(y_all).detach().cpu().numpy()
            pred = np.where(prob > 0.5, 1, 0)
            f1 = f1_score(label_all.detach().cpu().numpy(), pred)
            auc = roc_auc_score(label_all.detach().cpu().numpy(), prob)
            logger.info(f'F1-score for validation of epoch {epoch} is {f1:.3f}')
            logger.info(f'AUC for validation of epoch {epoch} is {auc:.3f}')
            acc_or_auc = auc
        else:
            pred = np.argmax(y_all.detach().cpu().numpy(), axis=1)
            acc = (pred == label_all.detach().cpu().numpy()).sum() / label_all.shape[0]
            logger.info(f'Acc for validation of epoch {epoch} is {acc:.3f}')
            acc_or_auc = acc
        logger.info(f'Epoch {epoch} validating finished !!!')
        return acc_or_auc