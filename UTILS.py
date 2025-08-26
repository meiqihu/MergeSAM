
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn as nn
import cv2
# from skimage.measure import label, regionprops
from skimage import measure
from pycocotools import mask as maskUtils
import json
import os
# from box import Box
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
from scipy.linalg import sqrtm
import os
import cv2
from skimage import exposure
import random
from scipy.sparse import csr_matrix

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        OA = (tp + tn) / (tp + fp + tn + fn)
        return OA

    def Kappa(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        PRE = ((tp + fp) * (tp + fn) + (tn + fn) * (fp + tn)) / ((tp + fp + tn + fn) * (tp + fp + tn + fn))
        OA = (tp + tn) / (tp + fp + tn + fn)
        Kappa = (OA - PRE) / (1 - PRE)
        return Kappa

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



def Pixel_evaluation(Eva_test, result_name,PixelEva=True):
    if PixelEva:
        IoU = Eva_test.Intersection_over_Union()
        Pre = Eva_test.Precision()
        Recall = Eva_test.Recall()
        F1 = Eva_test.F1()
        OA = Eva_test.OA()
        Kappa = Eva_test.Kappa()

        with open(result_name, 'a') as f:
            f.writelines('    (2) Pixel-level Evaluation: \n')
            f.write(f'F1: {F1[1]}\n')
            f.write(f'Pre: {Pre[1]}\n')
            f.write(f'Recall: {Recall[1]}\n')
            f.write(f'OA: {OA[1]}\n')
            f.write(f'Kappa: {Kappa[1]}\n')
            f.write(f'IoU: {IoU[1]}\n')
            f.write('\n')

        print('[Test] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
        print('F1-Score: Precision: Recall: OA: Kappa: IoU: ')
        print(
            '{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100,
                                                               Kappa[1] * 100, IoU[1] * 100))

# 假设 masks1 和 masks2 是二值化掩膜数组，形状为 (N, H, W)
# 转换为稀疏矩阵的稀疏格式
def convert_to_sparse(masks):
    # 每个掩膜是一个二维数组，形状为 (H, W)
    # 我们将掩膜展平为一维向量，然后转换为稀疏矩阵
    N, H, W = masks.shape
    masks_sparse = np.zeros((N, H * W), dtype=int)
    # 将每个掩膜展平并填充稀疏矩阵
    for i in range(N):
        masks_sparse[i] = masks[i].flatten()
    # 转换为稀疏矩阵格式（CSR）
    return csr_matrix(masks_sparse)
# 计算 IOU
def calculate_iou_sparse(masks1, masks2, threshold):
    masks1_sparse = convert_to_sparse(masks1)
    masks2_sparse = convert_to_sparse(masks2)
    intersection = np.dot(masks1_sparse, masks2_sparse.T)
    union = np.sum(masks1_sparse, axis=1).reshape(-1, 1) + np.sum(masks2_sparse, axis=1).reshape(1,
                                                                                                 -1) - intersection
    # 计算 IOU
    iou = intersection / union

    # 获取 IOU 大于 threshold 的有效匹配对的索引
    valid_matches = np.array(np.nonzero(iou > threshold)).T
    return valid_matches

def get_matched_and_remaining_masks(masks1, masks2, valid_matches_batch):
    """
    根据有效匹配的索引，获取对应的掩膜以及剩余的掩膜。

    Args:
        masks1_sparse: 稀疏矩阵，形状为 [N1, H*W]。
        masks2_sparse: 稀疏矩阵，形状为 [N2, H*W]。
        valid_matches_batch: 有效匹配对的索引列表，形状为 [num_matches, 2]。

    Returns:
        matched_masks1: 与 masks2 中匹配的掩膜 (形状：[num_matches, H*W])。
        matched_masks2: 与 masks1 中匹配的掩膜 (形状：[num_matches, H*W])。
        remaining_masks1: 未与 masks2 匹配的掩膜 (形状：[N1 - num_matches, H*W])。
        remaining_masks2: 未与 masks1 匹配的掩膜 (形状：[N2 - num_matches, H*W])。
    """
    # 获取有效匹配对的索引
    matched_indices1 = valid_matches_batch[:, 0]  # 从 masks1 中匹配的掩膜索引
    matched_indices2 = valid_matches_batch[:, 1]  # 从 masks2 中匹配的掩膜索引

    # 提取匹配的掩膜
    matched_masks1 = masks1[matched_indices1]
    matched_masks2 = masks2[matched_indices2]

    # 提取剩余的掩膜：取出未匹配的掩膜
    all_indices1 = np.arange(masks1.shape[0])  # 所有 masks1 的索引
    all_indices2 = np.arange(masks2.shape[0])  # 所有 masks2 的索引

    remaining_indices1 = np.setdiff1d(all_indices1, matched_indices1)  # 取出未匹配的 masks1 的索引
    remaining_indices2 = np.setdiff1d(all_indices2, matched_indices2)  # 取出未匹配的 masks2 的索引

    # 提取剩余的掩膜
    remaining_masks1 = masks1[remaining_indices1]
    remaining_masks2 = masks2[remaining_indices2]

    return matched_masks1, matched_masks2, remaining_masks1, remaining_masks2

def get_mergedmask(masks1, masks2):
    H, W = masks1[0].shape
    merged_mask = np.zeros((H, W))
    # 遍历所有掩膜并合并
    for i in range(len(masks1)):  # len(masks1)
        mask = masks1[i]
        mask = mask * (np.max(merged_mask) + 0.5)
        merged_mask = merged_mask + mask

    for i in range(len(masks2)):  # len(masks1)
        mask = masks2[i]
        mask = mask * (np.max(merged_mask) + 0.5)
        merged_mask = merged_mask + mask

    mask_value_list = np.unique(merged_mask)
    # # 存储最终的掩膜列表
    final_masks = []
    mask_area =[]
    # 遍历所有联通区域，跳过背景（标签0）
    for i in range(1, len(mask_value_list)):
        # 获取当前联通区域的掩膜
        current_mask = merged_mask == mask_value_list[i]
        # 将当前掩膜添加到最终掩膜列表
        area = np.sum(current_mask)
        mask_area.append(area)
        final_masks.append(current_mask)
    return final_masks


