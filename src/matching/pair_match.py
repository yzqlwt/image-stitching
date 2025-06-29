from __future__ import annotations

import cv2
import numpy as np

from src.images import Image


class PairMatch:
    def __init__(self, image_a: Image, image_b: Image, matches: list | None = None) -> None:
        """
        Create a new PairMatch object.

        Args:
            image_a: First image of the pair
            image_b: Second image of the pair
            matches: List of matches between image_a and image_b
        """
        self.image_a = image_a
        self.image_b = image_b
        self.matches = matches
        self.H = None
        self.status = None
        self.overlap = None
        self.area_overlap = None
        self._Iab = None
        self._Iba = None
        self.matchpoints_a = None
        self.matchpoints_b = None

    def compute_homography(
        self, ransac_reproj_thresh: float = 5, ransac_max_iter: int = 500
    ) -> None:
        """
        Compute the homography between the two images of the pair.

        Args:
            ransac_reproj_thresh: reprojection threshold used in the RANSAC algorithm
            ransac_max_iter: number of maximum iterations for the RANSAC algorithm
        """
        # self.matchpoints_a = np.float32(
        #     [self.image_a.keypoints[match.queryIdx].pt for match in self.matches]
        # )
        # self.matchpoints_b = np.float32(
        #     [self.image_b.keypoints[match.trainIdx].pt for match in self.matches]
        # )

        kpts0, kpts1, matches = self.image_a.keypoints, self.image_b.keypoints, self.matches

        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        self.matchpoints_a = m_kpts0.cpu().numpy()
        self.matchpoints_b = m_kpts1.cpu().numpy()

        self.H, self.status = cv2.findHomography(
            self.matchpoints_b,
            self.matchpoints_a,
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter,
        )

    def set_overlap(self) -> None:
        """Compute and set the overlap region between the two images."""
        if self.H is None:
            self.compute_homography()

        mask_a = np.ones_like(self.image_a.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(
            np.ones_like(self.image_b.image[:, :, 0], dtype=np.uint8), self.H, mask_a.shape[::-1]
        )

        self.overlap = mask_a * mask_b
        self.area_overlap = self.overlap.sum()

    def is_valid(self, alpha: float = 8, beta: float = 0.3) -> bool:
        """
        Check if the pair match is valid (i.e. if there are enough inliers with regard to the overlap region).

        Args:
            alpha: alpha parameter used in the comparison
            beta: beta parameter used in the comparison

        Returns:
            valid: True if the pair match is valid, False otherwise
        """
        if self.overlap is None:
            self.set_overlap()

        if self.status is None:
            self.compute_homography()

        matches_in_overlap = self.matchpoints_a[
            self.overlap[
                self.matchpoints_a[:, 1].astype(np.int64),
                self.matchpoints_a[:, 0].astype(np.int64),
            ]
            == 1
        ]

        return self.status.sum() > alpha + beta * matches_in_overlap.shape[0]

    def contains(self, image: Image) -> bool:
        """
        Check if the given image is contained in the pair match.

        Args:
            image: Image to check

        Returns:
            True if the given image is contained in the pair match, False otherwise
        """
        return self.image_a == image or self.image_b == image

    @property
    def Iab(self):
        if self._Iab is None:
            self.set_intensities()
        return self._Iab

    @Iab.setter
    def Iab(self, Iab):
        self._Iab = Iab

    @property
    def Iba(self):
        if self._Iba is None:
            self.set_intensities()
        return self._Iba

    @Iba.setter
    def Iba(self, Iba):
        self._Iba = Iba

    def set_intensities(self) -> None:
        """
        Compute the intensities of the two images in the overlap region.
        Used for the gain compensation calculation.
        """
        if self.overlap is None:
            self.set_overlap()

        inverse_overlap = cv2.warpPerspective(
            self.overlap, np.linalg.inv(self.H), self.image_b.image.shape[1::-1]
        )

        if self.overlap.sum() == 0:
            print(self.image_a.path, self.image_b.path)

        self._Iab = (
            np.sum(
                self.image_a.image * np.repeat(self.overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / self.overlap.sum()
        )
        self._Iba = (
            np.sum(
                self.image_b.image * np.repeat(inverse_overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / inverse_overlap.sum()
        )
