````markdown
# MergeSAM: Unsupervised Change Detection of Remote Sensing Images (IGARSS 2025)

**PyTorch code of the IGARSS 2025 paper:**  
**“MergeSAM: Unsupervised Change Detection of Remote Sensing Images Based on the Segment Anything Model (SAM)”**

- 论文（arXiv）：https://arxiv.org/abs/2507.22675  
- 会议：IGARSS 2025  
- 微信推送：https://mp.weixin.qq.com/s/EBGxqvBYl832S8MyHi9Q5w

---

## 一、项目简介

**Abstract**  
Recently, large foundation models trained on vast datasets have demonstrated exceptional capabilities in feature extraction and general feature representation. The ongoing advancements in deep learning-driven large models have shown great promise in accelerating unsupervised change detection methods, thereby enhancing the practical applicability of change detection technologies. Building on this progress, this paper introduces MergeSAM, an innovative unsupervised change detection method for high-resolution remote sensing imagery, based on the Segment Anything Model (SAM). Two novel strategies, MaskMatching and MaskSplitting, are designed to address real-world complexities such as object splitting, merging, and other intricate changes. The proposed method fully leverages SAM's object segmentation capabilities to construct multitemporal masks that capture complex changes, embedding the spatial structure of land cover into the change detection process.

---

## 二、方法流程与实验结果（来自论文）

> 请将论文中的“算法流程图”和“实验结果”图片加入仓库，并替换下方占位路径。

**算法流程图（示意）**  
![Algorithm Pipeline – MergeSAM](docs/figures/mergesam_pipeline.png)  
*Fig. 1. MergeSAM overall pipeline with **MaskMatching** & **MaskSplitting**.*

**实验结果（示意）**  
![Experimental Results – MergeSAM](docs/figures/mergesam_results.png)  
*Fig. 2. Qualitative/quantitative results on high-resolution change detection benchmarks.*

---

## 三、How to Start

> 以下以 `maincode.py` 为例（行号以你本地文件为准）。

1. **下载并引入 SAM 项目**
   - 官方仓库：https://github.com/facebookresearch/segment-anything  
   - 在 `maincode.py` **第 7 行**修改为你的本地路径（示例）：
     ```python
     # line 7
     SAM_PROJECT_PATH = "/absolute/path/to/segment-anything"
     import sys; sys.path.append(SAM_PROJECT_PATH)
     ```

2. **下载并设置 SAM 预训练权重**
   - 下载 SAM 预训练权重（如：`sam_vit_h_4b8939.pth`、`sam_vit_l_0b3195.pth`、`sam_vit_b_01ec64.pth`）。
   - 在 `maincode.py` **第 27、29、31 行**将 `sam_checkpoint` 替换为你的权重路径（示例）：
     ```python
     # lines 27/29/31 (examples)
     sam_checkpoint_h = "/absolute/path/to/sam_vit_h_4b8939.pth"
     sam_checkpoint_l = "/absolute/path/to/sam_vit_l_0b3195.pth"
     sam_checkpoint_b = "/absolute/path/to/sam_vit_b_01ec64.pth"
     ```

3. **下载数据并设置路径**
   - 下载 **GZ_CD_data** 或其他二值变化检测数据集，解压到本地。
   - 在 `maincode.py` **第 146 行**设置图像根目录（示例）：
     ```python
     # line 146
     img_root = "/absolute/path/to/GZ_CD_data"
     ```
   - **GZ_CD_data（百度网盘）**：https://pan.baidu.com/s/1TpeUDKIUH3iUXSsEe04YLg?pwd=359n  
     提取码：**359n**

---
````
