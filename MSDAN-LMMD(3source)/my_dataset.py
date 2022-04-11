# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as scio

random.seed(1)
txt_label = {"Normal": 0, "IR": 1, "OR": 2}


class filterdataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"Normal": 0, "IR": 1, "OR": 2}
        self.data_info = self.get_txt_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_txt, label = self.data_info[index]
        text = scio.loadmat(path_txt)
        text = np.array(text['originaldata'])  # originaldata  filterdata
        text = text.transpose()
        text = text.astype(np.float32)

        if self.transform is not None:
            text = self.transform(text)  # 在这里做transform，转为tensor等等

        return text, label

    def __len__(self):
        return len(self.data_info)

    # @staticmethod
    def get_txt_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                text_names = os.listdir(os.path.join(root, sub_dir))
                text_names = list(filter(lambda x: x.endswith('.mat'), text_names))

                # 遍历文档
                for i in range(len(text_names)):
                    text_name = text_names[i]
                    path_txt = os.path.join(root, sub_dir, text_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_txt, int(label)))

        return data_info
