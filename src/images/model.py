import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # 一定放在 import torch 之前
import os.path
import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch, batch_to_device
from lightglue import match_pair
from lightglue import viz2d
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
from src.images import Image


def getDevice():
    return torch.device("mps")


def getFeatures(image: Image):
    image = numpy_image_to_torch(image.image).to(getDevice())
    feats = extractor.extract(image.to(getDevice()))
    return feats


def match(image_a: Image, image_b: Image):
    matches01 = matcher({"image0": image_a.features, "image1": image_b.features})
    data = [image_a.features, image_b.features, matches01]
    feats0, feats1, matches01 = [batch_to_device(rbd(x)) for x in data]
    image_a.keypoints = feats0["keypoints"]
    image_b.keypoints = feats1["keypoints"]

    image0 = numpy_image_to_torch(image_a.image).to(getDevice())
    image1 = numpy_image_to_torch(image_b.image).to(getDevice())
    axes = viz2d.plot_images([image0, image1])
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.show()
    return matches01["matches"]


def match2(image_a: Image, image_b: Image):
    # 记录开始时间
    start_time = time.time()

    # 调用需要测量运行时间的函数
    image0 = numpy_image_to_torch(image_a.image).to(getDevice())
    image1 = numpy_image_to_torch(image_b.image).to(getDevice())
    feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
    image_a.features = feats0
    image_b.features = feats1
    image_a.keypoints = feats0["keypoints"]
    image_b.keypoints = feats1["keypoints"]

    # 记录结束时间
    end_time = time.time()

    # 计算函数运行的时间
    elapsed_time = end_time - start_time
    print(f"match:{os.path.basename(image_a.path)} {os.path.basename(image_b.path)} {elapsed_time:.2f} 秒")

    axes = viz2d.plot_images([image0, image1])
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    plt.show()

    return matches01["matches"]


# SuperPoint+LightGlue
# extractor = SuperPoint(max_num_keypoints=4096).eval().to(getDevice())  # load the extractor
# matcher = LightGlue(features='superpoint').eval().to(getDevice())  # load the matcher

# extractor = SIFT(max_num_keypoints=4096).eval().to(getDevice())  # load the extractor
# matcher = LightGlue(features='sift').eval().to(getDevice())  # load the matcher

extractor = DISK(max_num_keypoints=4096).eval().to(getDevice())  # load the extractor
matcher = LightGlue(features='disk').eval().to(getDevice())  # load the matcher




# # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
