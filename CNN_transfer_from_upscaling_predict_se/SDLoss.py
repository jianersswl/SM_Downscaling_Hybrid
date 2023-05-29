import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def display_downscaling(smap_downscaling):
    x = range(smap_downscaling.size(0))
    for i in range(smap_downscaling.size(1)):
        for j in range(smap_downscaling.size(2)):
            plt.plot(x, smap_downscaling[:, i, j])
    plt.show()
    
def display_smap(smap_list, smap_pred_list, ab_pred_list):
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
    
def display_sm(sm_list, sm_pred_list, ab_pred_list):
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

def self_defined_smap_loss(pred_ab, label_data, flag, penalty_constant):
    pred_a = pred_ab[:, 0].unsqueeze(1)
    pred_b = pred_ab[:, 1].unsqueeze(1)
    
    smap = label_data['smap']
    pred_smap, smap_downscaling = calculate_pred_smap(pred_ab, label_data)
    
    criterion = nn.MSELoss(reduction='mean')
    pred_smap_loss = criterion(pred_smap, smap)
    
    if flag=='Validing':
        display_smap(smap, pred_smap, pred_ab)
        display_downscaling(smap_downscaling)
    return pred_smap_loss, \
          less_equal_zero_loss(pred_a, penalty_constant['sd_lambda'], penalty_constant['sd_threshold']), \
          less_equal_zero_loss(smap_downscaling, penalty_constant['sm_lambda'], penalty_constant['sm_threshold']), \
          EDS_loss(pred_a, penalty_constant['eds_lambda'], penalty_constant['eds_threshold'])

def self_defined_sm_loss(pred_ab, label_data, flag, sim_threshold):
    pred_a = pred_ab[:, 0].unsqueeze(1)
    pred_b = pred_ab[:, 1].unsqueeze(1)
    
    sm, pred_sm, grid_loss = calculate_pred_sm(pred_ab, label_data)
    
    pred_sm_loss = torch.sum(grid_loss) / grid_loss.size(0)
    
    if flag=='Validing':
        display_sm(sm, pred_sm, pred_ab)

    return pred_sm_loss, less_equal_zero_loss(pred_a), less_equal_zero_loss(pred_sm)

def calculate_pred_smap(pred_ab, label_data): # sm = smap + a * (pij - p_bar) / p_sd + b
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
    
    # 创建一个与 ati_grid 形状相同的全零张量 pred_sm
    smap_downscaling = torch.zeros_like(ati_grid, dtype=torch.float32)
    
    # 计算 smap_downscaling，并将其保存在 pred_smap 中
    for i in range(pred_ab.size(0)):
        a, b = pred_ab[i]
        # 计算ati_grid[i]的均值
        ati_mean = torch.sum(ati_grid[i])/torch.count_nonzero(ati_grid[i])
        
        # 计算ati_grid[i]的标准差
        ati_sd = torch.sqrt(torch.sum(torch.mul(ati_grid[i] - ati_mean, ati_grid[i] - ati_mean))/torch.count_nonzero(ati_grid[i]))
        
        # 根据公式进行smap_downscaling的计算
        smap_downscaling[i] = smap[i] + a * (ati_grid[i]-ati_mean)/ati_sd + b
        smap_downscaling[i] = torch.mul(smap_downscaling[i], ati_valid[i]) 
        
        # 升尺度获得smap的预测值
        pred_smap[i] = torch.sum(smap_downscaling[i])/torch.count_nonzero(ati_grid[i])
    return pred_smap, smap_downscaling



def calculate_pred_sm(pred_ab, label_data):
    # 获取smap
    smap = label_data['smap']
    # 获取ati_grid
    ati_grid = label_data['ati_grid']
    
    # 获取insitu ati
    insitu_ati = label_data['insitu_ati']
    # 获取sm
    insitu_sm = label_data['insitu_sm']
    
    # 统计sm以获得pred_sm的shape
    sm = []
    for i in range(pred_ab.size(0)):
        for j in range(len(insitu_sm[i])):
            sm.append(insitu_sm[i][j][0])
            
    sm = torch.FloatTensor(sm)
    
    # 创建一个形状与 sm 相同的tensor
    pred_sm = torch.zeros_like(sm, dtype=torch.float32)
    
    # 创建一个形状与 smap 相同的tensor
    grid_loss = torch.zeros_like(smap, dtype=torch.float32)
    
    # 初始化指向pred_sm的index
    k = 0
    for i in range(pred_ab.size(0)):
        a, b = pred_ab[i]
        
        # 计算ati_grid[i]的均值
        ati_mean = torch.sum(ati_grid[i])/torch.count_nonzero(ati_grid[i])
        
        # 计算ati_grid[i]的标准差
        ati_sd = torch.sqrt(torch.sum(torch.mul(ati_grid[i] - ati_mean, ati_grid[i] - ati_mean))/torch.count_nonzero(ati_grid[i]))
        
        # 计算sm_downscaling和loss
        for j in range(len(insitu_sm[i])):
            sm_downscaling = smap[i] + a * (insitu_ati[i][j][0]-ati_mean) / ati_sd + b
            grid_loss[i] += (sm_downscaling-insitu_sm[i][j][0])*(sm_downscaling-insitu_sm[i][j][0])
            pred_sm[k] = sm_downscaling
            k += 1
        grid_loss[i] = grid_loss[i] / len(insitu_sm[i])
    return sm, pred_sm, grid_loss

def less_equal_zero_loss(pred, penalty_lambda, penalty_threshold):
    neg = torch.where(pred>0, 0, torch.pow(2, pred*-1))
#     if penalty_threshold > 0:
#         print('count of sm < 0: ', torch.count_nonzero(neg))
    neg_loss = torch.sum(neg)/penalty_lambda
    if neg_loss < penalty_threshold:
        neg_loss = 0 * neg_loss
    return neg_loss  
    
def EDS_loss(pred, penalty_lambda, penalty_threshold):
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