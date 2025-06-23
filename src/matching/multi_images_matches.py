import logging
import cv2
import numpy as np
from src.images import Image
from src.images.model import match
from src.matching.pair_match import PairMatch


class MultiImageMatches:
    def __init__(self, images: list[Image], ratio: float = 0.75) -> None:
        """
        Create a new MultiImageMatches object.

        Args:
            images: images to compare
            ratio: ratio used for the Lowe's ratio test
        """
        self.images = images
        self.matches = {image.path: {} for image in images}
        self.ratio = ratio

    def get_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Get matches for the given images.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: List of matches between the two images
        """
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches

        return self.matches[image_a.path][image_b.path]

    # def get_pair_matches(self, max_images: int = 6) -> list[PairMatch]:
    #     """
    #     Get the pair matches for the given images.
    #
    #     Args:
    #         max_images: Number of matches maximum for each image
    #
    #     Returns:
    #         pair_matches: List of pair matches
    #     """
    #     pair_matches = []
    #     for i, image_a in enumerate(self.images):
    #         possible_matches = sorted(
    #             self.images[:i] + self.images[i + 1:],
    #             key=lambda image, ref=image_a: len(self.get_matches(ref, image)),
    #             reverse=True,
    #         )[:max_images]
    #         for image_b in possible_matches:
    #             if self.images.index(image_b) > i:
    #                 pair_match = PairMatch(image_a, image_b, self.get_matches(image_a, image_b))
    #                 if pair_match.is_valid():
    #                     pair_matches.append(pair_match)
    #     return pair_matches
    def get_pair_matches(self, max_images: int = 6) -> list[PairMatch]:
        """
        Get the pair matches for the given images.

        Args:
            max_images: Number of matches maximum for each image

        Returns:
            pair_matches: List of pair matches
        """
        pair_matches = []
        # 遍历每张图片
        for i, image_a in enumerate(self.images[:-1]):  # 最后一张图像不需要与后面的图像匹配
            # 仅匹配当前图像与后面一张图像
            image_b = self.images[i + 1]

            # 获取当前图像与 image_b 之间的匹配点
            matches = self.get_matches(image_a, image_b)

            # 创建配对对象
            pair_match = PairMatch(image_a, image_b, matches)

            # 如果匹配有效，则加入列表
            if pair_match.is_valid(beta=0.1):
                pair_matches.append(pair_match)

        return pair_matches

    def compute_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Compute matches between image_a and image_b.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: Matches between image_a and image_b
        """
        matches = match(image_a, image_b)

        return matches
