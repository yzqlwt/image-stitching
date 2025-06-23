"""
Main script.
Create panoramas from a set of images.
"""

import argparse
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np

from src.images import Image
from src.matching import (
    MultiImageMatches,
    PairMatch,
    build_homographies,
    find_connected_components,
)
from src.rendering import multi_band_blending, set_gain_compensations, simple_blending

start_time = time.time()

parser = argparse.ArgumentParser(
    description="Create panoramas from a set of images. \
                 All the images must be in the same directory. \
                 Multiple panoramas can be created at once"
)

parser.add_argument(dest="data_dir", type=Path, help="directory containing the images")
parser.add_argument(
    "-mbb",
    "--multi-band-blending",
    default=False,
    action="store_true",
    help="use multi-band blending instead of simple blending",
)
parser.add_argument(
    "--size", type=int, help="maximum dimension to resize the images to"
)
parser.add_argument(
    "--num-bands", type=int, default=5, help="number of bands for multi-band blending"
)
parser.add_argument(
    "--mbb-sigma", type=float, default=1, help="sigma for multi-band blending"
)

parser.add_argument(
    "--gain-sigma-n", type=float, default=10, help="sigma_n for gain compensation"
)
parser.add_argument(
    "--gain-sigma-g", type=float, default=0.1, help="sigma_g for gain compensation"
)

parser.add_argument(
    "-v", "--verbose", action="store_true", help="increase output verbosity"
)

args = vars(parser.parse_args())

logging.basicConfig(
    level=logging.INFO,  # 根据 -v 参数设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式：时间 - 日志级别 - 消息
    datefmt="%Y-%m-%d %H:%M:%S",  # 日期格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("panorama_creation.log", mode="w", encoding="utf-8"),  # 输出到文件
    ],
)

logging.info("Gathering images...")

valid_images_extensions = {".jpg", ".png", ".bmp", ".jpeg"}

image_paths = [
    str(filepath)
    for filepath in args["data_dir"].iterdir()
    if filepath.suffix.lower() in valid_images_extensions
]

# image_paths = sorted(image_paths, key=lambda x: int(re.search(r'\d+', x).group()))
image_paths = sorted(image_paths)
print(image_paths)

images = [Image(path, args.get("size")) for path in image_paths]

logging.info("Found %d images", len(images))
logging.info("Computing features with SIFT...")

for image in images:
    image.compute_features()

logging.info("Matching images with features...")

matcher = MultiImageMatches(images)
pair_matches: list[PairMatch] = matcher.get_pair_matches()
pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)

logging.info("Finding connected components...")
logging.info("Total pair matches received: %d", len(pair_matches))
connected_components = find_connected_components(pair_matches)

logging.info("Found %d connected components", len(connected_components))
logging.info("Building homographies...")

build_homographies(connected_components, pair_matches)

time.sleep(0.1)

logging.info("Computing gain compensations...")

for connected_component in connected_components:
    component_matches = [
        pair_match
        for pair_match in pair_matches
        if pair_match.image_a in connected_component
    ]

    set_gain_compensations(
        connected_component,
        component_matches,
        sigma_n=args["gain_sigma_n"],
        sigma_g=args["gain_sigma_g"],
    )

time.sleep(0.1)

for image in images:
    image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

results = []

if args["multi_band_blending"]:
    logging.info("Applying multi-band blending...")
    results = [
        multi_band_blending(
            connected_component,
            num_bands=args["num_bands"],
            sigma=args["mbb_sigma"],
        )
        for connected_component in connected_components
    ]


else:
    logging.info("Applying simple blending...")
    results = [
        simple_blending(connected_component)
        for connected_component in connected_components
    ]
if len(results) > 0:
    logging.info("Saving results to %s", args["data_dir"] / "results")
else:
    logging.info("No results to save")

(args["data_dir"] / "results").mkdir(exist_ok=True, parents=True)

for i, result in enumerate(results):
    cv2.imwrite(str(args["data_dir"] / "results" / f"pano_{i}.jpg"), result)

end_time = time.time()

# 计算函数运行的时间
elapsed_time = end_time - start_time
print(f"total cost: {elapsed_time:.2f} 秒")
