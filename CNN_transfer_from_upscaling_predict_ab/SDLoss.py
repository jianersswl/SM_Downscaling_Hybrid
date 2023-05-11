import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def display_smap(smap_list, smap_pred_list, ab_pred_list, flag):
    x = range(len(smap_list))
    ytick1 = torch.arange(0,1,0.1)
    ytick2 = torch.arange(-5,5,1)
    
    # 创建子图
    fig, (fig_ax1, fig_ax2) = plt.subplots(1, 2, figsize=(10, 2.5), sharey=False)

    # 设置第一个子图的标题和横纵坐标标签
    fig_ax1.set_title('SMAP')
    fig_ax1.set_xlabel('X-axis')
    fig_ax1.set_ylabel('Y-axis')
#     fig_ax1.set_yticks(ytick1)

    # 绘制第一个子图
    fig_ax1.plot(x, smap_list, label='smap')
    fig_ax1.plot(x, smap_pred_list, label='smap_pred')
    fig_ax1.legend()
    
    # 设置第二个子图的标题和横纵坐标标签
    fig_ax2.set_title('Model Prediction')
    fig_ax2.set_xlabel('X-axis')
    fig_ax2.set_ylabel('Y-axis')
#     fig_ax2.set_yticks(ytick2)
    
    # 绘制第二个子图
    fig_ax2.plot(x, ab_pred_list[:, 0], label='a')
    fig_ax2.plot(x, ab_pred_list[:, 1], label='b')
    fig_ax2.legend()
    
    # 调整子图布局，避免重叠
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
def display_sm(sm_list, sm_pred_list, ab_pred_list, flag):
    x1 = range(len(sm_list))
    x2 = range(len(ab_pred_list))
    ytick1 = torch.arange(0,1,0.1)
    ytick2 = torch.arange(-5,5,1)
    
    # 创建子图
    fig, (fig_ax1, fig_ax2) = plt.subplots(1, 2, figsize=(10, 2.5), sharey=False)

    # 设置第一个子图的标题和横纵坐标标签
    fig_ax1.set_title('SM')
    fig_ax1.set_xlabel('X-axis')
    fig_ax1.set_ylabel('Y-axis')
#     fig_ax1.set_yticks(ytick1)

    # 绘制第一个子图
    fig_ax1.plot(x1, sm_list, label='sm')
    fig_ax1.plot(x1, sm_pred_list, label='sm_pred')
#     fig_ax1.plot(x1, sm_list-sm_pred_list, label='sm_pred')
    fig_ax1.legend()
    
    # 设置第二个子图的标题和横纵坐标标签
    fig_ax2.set_title('Model Prediction')
    fig_ax2.set_xlabel('X-axis')
    fig_ax2.set_ylabel('Y-axis')
#     fig_ax2.set_yticks(ytick2)
    
    # 绘制第二个子图
    fig_ax2.plot(x2, ab_pred_list[:, 0], label='a')
    fig_ax2.plot(x2, ab_pred_list[:, 1], label='b')
    fig_ax2.legend()
    
    # 调整子图布局，避免重叠
    plt.tight_layout()
    
    # 显示图形
    plt.show()

def self_defined_sm_loss(pred_ab, label_data, flag, sim_threshold):
    pred_a = pred_ab[:, 0].unsqueeze(1)
    pred_b = pred_ab[:, 1].unsqueeze(1)
    return ab_physics_loss_sm(pred_ab, label_data, flag), EDS_loss(pred_a, sim_threshold, 1) #+ EDS_loss(pred_b, sim_threshold, 0.5)

def self_defined_smap_loss(pred_ab, label_data, flag, sim_threshold):
    pred_a = pred_ab[:, 0].unsqueeze(1)
    pred_b = pred_ab[:, 1].unsqueeze(1)
    return ab_physics_loss_smap(pred_ab, label_data, flag), EDS_loss(pred_a, sim_threshold, 10) #+ EDS_loss(pred_b, sim_threshold, 1)

def ab_physics_loss_smap(pred_ab, label_data, flag): # sm = a * pij + b
    # 获取smap
    smap = label_data['smap']
    # 获取ati_grid
    ati_grid = label_data['ati_grid']

    # 创建一个形状与 ati_grid 中二维张量相同的二值张量 ati_valid
    ati_valid = torch.zeros_like(ati_grid, dtype=torch.float32)

    # 遍历 ati_grid 中的每个二维张量，将其转换成对应的二值矩阵
    for i in range(ati_valid.size(0)):
        ati_valid[i] = (ati_grid[i] != 0).type(torch.float32)
    
    # 创建一个与 smap 形状相同的全零张量 pred_smap
    pred_smap = torch.zeros_like(smap, dtype=torch.float32)
    # 计算 smap_downscaling，并将其保存在 pred_smap 中
    for i in range(pred_ab.size(0)):
        a, b = pred_ab[i]
        smap_downscaling = torch.mul(a * ati_grid[i] + b, ati_valid[i])
        pred_smap[i] = torch.sum(smap_downscaling)/torch.nonzero(ati_valid[i]).size(0)
    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(pred_smap, smap)
    if flag=='Validing':
        display_smap(smap, pred_smap, pred_ab, flag)
    return loss

def ab_physics_loss_sm(pred_ab, label_data, flag): # sm = a * pij + b
    # 获取smap
    smap = label_data['smap']
    # 获取ati
    insitu_ati = label_data['insitu_ati']
    # 获取sm
    insitu_sm = label_data['insitu_sm']
    
    # 可视化数据
    sm = []
    pred_sm = []
    
    # 创建一个形状与 pred_ab 相同的loss张量
    loss = torch.zeros_like(smap.squeeze(1), dtype=torch.float32)
    # 计算loss张量
    for i in range(pred_ab.size(0)):
        a, b = pred_ab[i]
        for j in range(len(insitu_sm[i])):
            sm_downscaling = a * insitu_ati[i][j][0] + b
            loss[i] += (sm_downscaling-insitu_sm[i][j][0])*(sm_downscaling-insitu_sm[i][j][0])
            sm.append(insitu_sm[i][j][0])
            pred_sm.append(sm_downscaling)
        loss[i] = loss[i] / len(insitu_sm[i])
    # 计算MSELoss
    mseloss = torch.sum(loss) / loss.size(0)
    
    if flag=='Validing':
        display_sm(sm, pred_sm, pred_ab, flag)
    return mseloss

def cos_sim_loss(pred_ab, penalty_lambda):
    vec1 = pred_ab.unsqueeze(0)
    vec2 = pred_ab.unsqueeze(1)
    cos_sim_mat = F.cosine_similarity(vec1, vec2, dim=-1)
    
    # 获取对角线上的值
    diag = torch.diag(cos_sim_mat)
    # 获取非对角线上的值
    non_dig_sum = torch.sum(cos_sim_mat - torch.diag(diag))
    # 计算均值
    non_dig_sum_mean = non_dig_sum/(cos_sim_mat.shape[0]*(cos_sim_mat.shape[0]-1))
    
    if non_dig_sum_mean<0:
        penalty_lambda = 0
    loss = penalty_lambda*non_dig_sum_mean*non_dig_sum_mean**non_dig_sum_mean
    return loss
    
def EDS_loss(pred, penalty_threshold, penalty_lambda):
    eds_sim_mat = torch.cdist(pred, pred)
    # 获取对角线上的值
    diag = torch.diag(eds_sim_mat)
    # 获取非对角线上的值
    non_dig_sum = torch.sum(eds_sim_mat - torch.diag(diag))
    # 计算均值
    non_dig_sum_mean = non_dig_sum/(eds_sim_mat.shape[0]*(eds_sim_mat.shape[0]-1))
    similarity = 1/(1+non_dig_sum_mean)
#     print('similarity: ', eds_sim_mat)
#     similarity = similarity*similarity
    if(similarity<=penalty_threshold):
        penalty_lambda = 0
    loss = penalty_lambda*similarity
#     print('loss: ', loss)
    return loss    