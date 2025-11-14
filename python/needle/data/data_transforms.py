from typing import Any

import numpy as np


class Transform:
    def __call__(self, x: Any) -> Any:
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding: int = 3) -> None:
        self.padding = padding

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        # pad all sides of image
        padded_img = np.pad(
            img,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
        )
        # crop image
        cropped_img = padded_img[
            self.padding + shift_x : self.padding + shift_x + img.shape[0],
            self.padding + shift_y : self.padding + shift_y + img.shape[1],
            :,
        ]
        return cropped_img
