import numpy as np
import torch
from torch import nn


def CoxLoss(hazard_pred, label, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    survtime, censor = label[:, 0], label[:, 1]
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def Knowledge_decomposition(cspx,gspx,wspx,cgshx,gwshx,wcshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share):
    cos = nn.CosineSimilarity(dim=1)
    cos1 = cos(cspx, gspx).abs().mean() + cos(gspx, wspx).abs().mean() + cos(wspx, cspx).abs().mean()
    cos2 = cos(cspx, cgshx).abs().mean() + cos(gspx, cgshx).abs().mean() + cos(gspx, gwshx).abs().mean() + cos(wspx, gwshx).abs().mean() + cos(wspx, wcshx).abs().mean()+ cos(cspx, wcshx).abs().mean()
    cos3 = cos(dualx1, dualx2).abs().mean() + cos(dualx2, dualx3).abs().mean() + cos(dualx3, dualx1).abs().mean()
    cos4 = (1 - cos(triple_share1, triple_share2)).mean() + (1 - cos(triple_share2, triple_share3)).mean() + (
                1 - cos(triple_share3, triple_share1)).mean()
    cos5 = cos(dualx1, triple_share).abs().mean() + cos(dualx2, triple_share).abs().mean() + cos(dualx3, triple_share).abs().mean()
    return cos1 + cos2 + cos3 + cos4 + cos5


class survivalTimeLoss(nn.Module):
    def __init__(self):
        super(survivalTimeLoss, self).__init__()

    def forward(self, predict_time, label):
        true_time, censored = label[:, 0], label[:, 1]
        predict_time = torch.exp(predict_time).reshape(-1,)
        # Calculate mae loss
        mae_loss = torch.abs(predict_time - true_time)
        # max(predict_time - true_time, 0)
        max_loss = torch.clamp(true_time-predict_time, min=0)
        loss = torch.where(censored == 1, mae_loss, max_loss)
        return loss.mean()


def survival_loss(predict, label):
    stLoss = survivalTimeLoss()
    return stLoss(predict, label)