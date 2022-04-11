#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
from torch.autograd import Variable
def convert_to_onehot(sca_label, class_num=7):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=8, class_num=7):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)             #np.sum(s_vec_label, axis=0) 代表每列（axis=0）元素求和
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = np.zeros((batch_size, batch_size))      #权重矩阵初始为0
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)           #   [:,i] 取所有行的第i列数据     (batch-size行，1列)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)                                   #相乘变成一个矩阵
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1
                # weight_st = weight(s_label,s_prelabel);
                # count += 1;

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def lmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual')
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    return loss