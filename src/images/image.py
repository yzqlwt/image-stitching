from __future__ import annotations
import cv2
import numpy as np


class Image:
    def __init__(self, path: str, size: int | None = None) -> None:
        """
        Image constructor.

        Args:
            path: path to the image
            size: maximum dimension to resize the image to
        """
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        if size is not None:
            h, w = self.image.shape[:2]
            if max(w, h) > size:
                if w > h:
                    self.image = cv2.resize(self.image, (size, int(h * size / w)))
                else:
                    self.image = cv2.resize(self.image, (int(w * size / h), size))

        self.features = None
        self.keypoints = None
        self.H: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)

    def compute_features(self) -> None:
        """Compute the features and the keypoints of the image using SIFT."""
        # image = numpy_image_to_torch(self.image).to(getDevice())
        # self.features = getExtractor().extract(image)
        from .model import getFeatures
        self.features = getFeatures(self)
        pass
