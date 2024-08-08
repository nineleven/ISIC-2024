import cv2
import numpy as np
import random


class ImageAugmenter:
    def random_rotation_flip(self, image):
        """
        Apply random rotation and random horizontal flip to the image.

        :param image: Input image as a numpy array (HxWxC)
        :return: Augmented image as a numpy array of the same shape
        """
        # Random rotation
        angle = random.uniform(-180, 180)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))

        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)

        return image

    def random_shear(self, image):
        """
        Apply random shear to the image.

        :param image: Input image as a numpy array (HxWxC)
        :return: Augmented image as a numpy array of the same shape
        """
        h, w = image.shape[:2]
        shear_factor = random.uniform(-0.3, 0.3)
        M = np.array([[1, shear_factor, 0],
                      [0, 1, 0]], dtype=np.float32)
        nW = w + abs(shear_factor * h)
        image = cv2.warpAffine(image, M, (int(nW), h))

        # Crop to original size
        if shear_factor > 0:
            image = image[:, :w]
        else:
            image = image[:, int(abs(shear_factor * h)):]

        return image

    def random_scale(self, image):
        """
        Apply random scale to the image.

        :param image: Input image as a numpy array (HxWxC)
        :return: Augmented image as a numpy array of the same shape
        """
        h, w = image.shape[:2]
        scale_factor = random.uniform(0.7, 1.3)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        # Crop or pad to the original size
        if scale_factor < 1.0:
            pad_w = (w - new_size[0]) // 2
            pad_h = (h - new_size[1]) // 2
            image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            start_x = (new_size[0] - w) // 2
            start_y = (new_size[1] - h) // 2
            image = image[start_y:start_y + h, start_x:start_x + w]

        return image

# Example usage:
# augmenter = ImageAugmenter()
# image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
# augmented_image = augmenter.random_rotation_flip(image)
# augmented_image = augmenter.random_shear(image)
# augmented_image = augmenter.random_scale(image)
