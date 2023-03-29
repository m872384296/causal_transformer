from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score

def train_one_epoch(config, rank, epoch, net, loss_fn, train_loader, optimizer, lr_scheduler, scaler, logger, writer):
    logger.info(f'Epoch {epoch} begin training......')
    net.train()
    if rank == 0:
        iterator = tqdm(train_loader, desc=f'Training epoch {epoch}')
    else:
        iterator = train_loader
    num_steps = len(train_loader)
    label_all = torch.tensor([]).cuda(rank)
    y_all = torch.tensor([]).cuda(rank)
    loss_all = 0
    for n_iter, (img, label, conf, idx) in enumerate(iterator):
        img = img.cuda(rank, non_blocking=True)
        label = label.cuda(rank, non_blocking=True)
        conf = conf.cuda(rank, non_blocking=True)
        idx = idx.cuda(rank, non_blocking=True)
        with autocast():
            y, y_feature = net[0](img)
            y_bar = net[1](conf, y_feature.detach())
            soft_split_all = net[2].module.soft_split
            split_eval = soft_split_all.detach()
            # loss1, erm1, penalty1 = loss_fn(y, label, split_eval, idx)
            # loss2, erm2, penalty2 = loss_fn(y_bar, label, split_eval, idx)
            # loss3, erm3, penalty3 = loss_fn(y_bar.detach(), label, soft_split_all, idx)
            # loss = loss1 - loss2 - loss3
            loss = loss_fn(y, label, split_eval, idx)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_reduce = loss.item() / config['world_size']
        loss_all += loss.item() * label.shape[0]
        if rank == 0:
            writer.add_scalar('Learning rate', lr, epoch * num_steps + n_iter)
            writer.add_scalar('Loss', loss_reduce, epoch * num_steps + n_iter)
        y_out = torch.zeros(config['world_size'] * y.shape[0], y.shape[1]).cuda(rank)
        dist.all_gather_into_tensor(y_out, y.float())
        y_all = torch.cat((y_all, y_out), 0)
        label_out = torch.zeros(config['world_size'] * label.shape[0]).cuda(rank)
        dist.all_gather_into_tensor(label_out, label.float())
        label_all = torch.cat((label_all, label_out), 0)
        if loss_fn.num_classes == 1:
            prob = torch.sigmoid(y_all).detach().cpu().numpy()
            pred = np.where(prob > 0.5, 1, 0)
            f1 = f1_score(label_all.detach().cpu().numpy(), pred)
            auc = roc_auc_score(label_all.detach().cpu().numpy(), prob)
            if rank == 0:
                writer.add_scalar('F1-score', f1, epoch * num_steps + n_iter)
                writer.add_scalar('AUC', auc, epoch * num_steps + n_iter)
        else:
            pred = np.argmax(y_all.detach().cpu().numpy(), axis=1)
            acc = (pred == label_all.detach().cpu().numpy()).sum() / label_all.shape[0]
            if rank == 0:
                writer.add_scalar('ACC', acc, epoch * num_steps + n_iter)
    loss_mean = loss_all / len(train_loader.dataset)
    if loss_fn.num_classes == 1:
        logger.info(f'F1-score for training of epoch {epoch} is {f1:.3f}')
        logger.info(f'AUC for training of epoch {epoch} is {auc:.3f}')
    else:
        logger.info(f'Acc for training of epoch {epoch} is {acc:.3f}')
    logger.info(f'Loss of epoch {epoch} is {loss_mean:.3f}')
    logger.info(f'Epoch {epoch} training finished !!!')
        
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
            logger.info(f'F1-score for validation of epoch {epoch} is {f1:.3f}')
            logger.info(f'AUC for validation of epoch {epoch} is {auc:.3f}')
            acc_or_auc = auc
        else:
            logger.info(f'Acc for validation of epoch {epoch} is {acc:.3f}')
            acc_or_auc = acc
        logger.info(f'Epoch {epoch} validating finished !!!')
        return acc_or_auc