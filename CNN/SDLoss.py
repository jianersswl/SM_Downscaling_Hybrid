import torch
import torch.nn as nn

def calculate_upscaling_sm(sm_bar, sm_bar_sd, ati, ati_bar, ati_bar_sd):
    return sm_bar + sm_bar_sd * (ati - ati_bar) / ati_bar_sd

def physics_loss(pred, label_data, batch_size):
    criterion = nn.MSELoss(reduction='mean')
    loss = 0
    for i, y in enumerate(pred):      # each calculation in a batch
        # calculate the high resolution sm for each in situ
#         print(label_data[i])
        for situ_pkg in label_data[i]: # each calculation in a smap
            sm_situ = situ_pkg[0][0]
            sm_bar = situ_pkg[1][0]
            ati =  situ_pkg[2][0]
            atim = situ_pkg[2][1]
            atisd = situ_pkg[2][2]
            sm_pred = calculate_upscaling_sm(sm_bar, y, ati, atim, atisd)
#             _loss = criterion(sm_pred, torch.FloatTensor([sm_situ]))
            _loss = criterion(sm_pred, torch.tensor(sm_situ, dtype=torch.float32))
            loss = (loss + _loss)
    loss = loss / batch_size
    return loss