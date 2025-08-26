
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
# sys.path.append("..")
sys.path.append("/media/home/xxx/segment-anything/")   # Please change the path here !!!
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# from utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tifffile
np.random.seed(3)
from UTILS import Evaluator, Pixel_evaluation, calculate_iou_sparse
from UTILS import get_matched_and_remaining_masks,get_mergedmask

# 测试1: 以IOU>0.95的初步筛选掉粗略的未变化对象,后续再参加阈值分割;利用merge obj产生多时相对象
def maincode(img_root, result_root, data_name, model_type = "vit_b", IoU_threshold=0.75, PixelEva=True):
    if model_type == "vit_b":
        sam_checkpoint = './SAM_pth/sam_vit_b_01ec64.pth'  # Please change the path here !!!
    elif model_type == "vit_l":
        sam_checkpoint = './SAM_pth/sam_vit_l_0b3195.pth'  # Please change the path here !!!
    elif model_type == "vit_h":
        sam_checkpoint = './SAM_pth/sam_vit_h_4b8939.pth' # Please change the path here !!!


    threshold_value = 155  # fixed threshold
    Eva_test = Evaluator(num_class=2)
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.8,
        box_nms_thresh=0.7)
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(os.path.join(result_root, 'Bmap'), exist_ok=True)
    os.makedirs(os.path.join(result_root, 'Changescore'), exist_ok=True)
    files_img1 = os.listdir(os.path.join(img_root, 'A'))
    files_img1 = sorted(files_img1)
    for file in tqdm(files_img1):
        filename1 = os.path.join(img_root, 'A', file)
        filename2 = os.path.join(img_root, 'B', file)
        filename_gt = os.path.join(img_root, 'label', file)
        save_map_path = os.path.join(result_root, 'Bmap', file)

        if os.path.exists(save_map_path):
            Bimap = cv2.imread(save_map_path, cv2.IMREAD_GRAYSCALE)
            pred = (Bimap / 255).astype(int)
        else:
            change_scores, pred = MergeSAM(filename1, filename2, mask_generator, result_root, IoU_threshold)

        target = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)
        if data_name == 'GZ_CD_resize1600':
            target = (target / 255).astype(int)  # for LIVIRCD
        Eva_test.add_batch(target, pred)

    result_name = os.path.join(result_root, data_name + '_EvaResults.txt')
    with open(result_name, 'a+') as f:
        f.write('\n')
        f.write(f'data_name: {data_name}\n')
        f.write(f'SAM model_type: {model_type}\n')
        f.write(f'threshold_value: {threshold_value}\n')

    if PixelEva:
        Pixel_evaluation(Eva_test, result_name, PixelEva=True)
    return pred

def MergeSAM(filename1, filename2, mask_generator, result_root, IoU_threshold):
    image1 = cv2.imread(filename1)  # [256, 256, 3]
    image2 = cv2.imread(filename2)  # [256, 256, 3]

    orig_H, orig_W = image1.shape[:2]
    masks1, _ = mask_generator.generate_onlyMask_Embed(image1)
    masks2, _ = mask_generator.generate_onlyMask_Embed(image2)

    file = os.path.basename(filename1)
    img_emb1 = image1.astype('float')
    img_emb2 = image2.astype('float')

    # 以IOU>0.75的初步筛选掉粗略的未变化对象,后续再参加阈值分割
    valid_matches = calculate_iou_sparse(masks1, masks2, IoU_threshold)
    match_masks1, match_masks2, remaining_masks1, remaining_masks2 = get_matched_and_remaining_masks(masks1,masks2,
                                                                                                 valid_matches)

    merged_mask = get_mergedmask(remaining_masks1, remaining_masks2)  # list all masks
    if len(valid_matches)>0:
        match_masks1 = match_masks1.tolist()
        match_masks2 = match_masks2.tolist()
        merged_mask = merged_mask + match_masks1 + match_masks2


    change_score_masks = []
    Changescore = np.zeros((orig_H, orig_W))
    save_map_path = os.path.join(result_root, 'Changescore', file)
    for mask_area in merged_mask:
        avg_features = img_emb1[mask_area, :]  # 只保留掩膜区域的像素，结果尺寸为 [C, N]，其中 N 是掩膜区域的像素数
        avg_features1 = avg_features.mean(axis=0)  # 结果为 [C]，每个通道的平均值

        avg_features = img_emb2[mask_area, :]  # 只保留掩膜区域的像素，结果尺寸为 [C, N]，其中 N 是掩膜区域的像素数
        avg_features2 = avg_features.mean(axis=0)  # 结果为 [C]，每个通道的平均值
        score = np.sum((avg_features1 - avg_features2) ** 2) # root of MSE
        score = np.sqrt(score)
        change_score_masks.append([score, mask_area])

    for score, mask in change_score_masks:
        Changescore[mask] = score  # 1-cos()

    plt.imshow(Changescore)
    plt.savefig(save_map_path)
    plt.close()

    F_min = Changescore.min()
    F_max = Changescore.max()
    F_stretched = (Changescore - F_min) / (F_max - F_min) * 255
    # 确保结果在 [0, 255] 范围内，并转换为 uint8 类型
    F_stretched = np.clip(F_stretched, 0, 255).astype('uint8')
    score_stretched = np.around(F_stretched).astype('uint8')
    threshold_value_otsu, binary_image = cv2.threshold(score_stretched, 0, 255,
                                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    save_map_path = os.path.join(result_root, 'Bmap', file)
    cv2.imwrite(save_map_path, binary_image)

    binary_image = (binary_image / 255).astype(int)
    return change_score_masks, binary_image

if __name__=='__main__':
    #------Before your Start----------
    # (1) Please download the Segment anything model project from https://github.com/facebookresearch/segment-anything, and
    # change the path in line 7;
    # (2) Plase download the pretrained weight of SAM model, and replace the "sam_checkpoint" with your path in line 27,29,31.
    # (3) Please download the GZ_CD_data or other binary change detection dataset, and replace the "img_root" with your path in line 146;
    # (4) Please replace the "result_root" with your path in line 147;
    data_name = 'GZ_CD_resize1600'
    model_type = "vit_b"  # "vit_l", "vit_h"
    img_root = './data/CD_Data_GZ/'  # Please change the path here !!!
    result_root = './result/'  # Please change the path here !!!
    result_root = os.path.join(result_root, model_type)

    maincode(img_root, result_root, data_name, model_type)

