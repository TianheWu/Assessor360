from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

import numpy as np
import logging
import torch


def train_oiqa(epoch, net, criterion, optimizer, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        d = data['d_img_org'].cuda()
        labels = data['score']
        name = data['name']

        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        pred_d = net(d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(
        epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_oiqa(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            d = data['d_img_org'].cuda()
            labels = data['score']
            name = data['name']

            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            pred_d = net(d)

            # compute loss
            loss = criterion(torch.squeeze(pred_d), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred_d.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(
                epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p