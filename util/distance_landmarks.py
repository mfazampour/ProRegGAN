import numpy as np
import torch


def get_index(lmark_truth, lmark_pred, number):

    index_t = torch.where(lmark_truth == number)
    index_p = torch.where(lmark_pred == number)

  #  print(f" number: {number} \n index_t: {index_t}  \n index_p: {index_p}")
    #
    if len(index_p[0]) < 1:
        if torch.mean(index_t[0].float()) > 41:
            ind_x = 90
        else:
            ind_x = - 10
        if torch.mean(index_t[1].float()) > 41:
            ind_y = 90
        else:
            ind_y = - 10
        if torch.mean(index_t[2].float()) > 41:
            ind_z = 90
        else:
            ind_z = - 10
        index_p = [torch.tensor([ind_x]), torch.tensor([ind_y]), torch.tensor([ind_z])]
    return index_t, index_p


#

def get_distance_lmark(lmark_truth, lmark_pred, device):
    landmark_tot_distance = []

    for i in range(len(lmark_truth)):

        for landmark in range(int(torch.max(lmark_truth))):
            index_t, index_p = get_index(lmark_truth[i, :, :, :], lmark_pred[i, :, :, :], landmark + 1)

            if len(index_t) != len(index_p):
                continue
            diff = torch.stack(index_t)[1:4, :].float().mean(dim=1) - torch.stack(index_p)[1:4, :].float().mean(dim=1)

            landmark_tot_distance.append((diff ** 2).sum().sqrt().to(device))

    if len(landmark_tot_distance) == 0:
        return torch.tensor([0.0]).to(device)
    else:
        return torch.mean(torch.stack(landmark_tot_distance))
