import albumentations as A


class ImageAugmenter:
    def __init__(self):
        self.hflip = A.HorizontalFlip(p=0.5)
        self.affine = A.Affine(rotate=(-180, 180), scale=(0.75, 1.25), shear=(-20, 20), p=0.5)


# Example usage:
# augmenter = ImageAugmenter()
# image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
# augmented_image = augmenter.random_rotation_flip(image)
# augmented_image = augmenter.random_shear(image)
# augmented_image = augmenter.random_scale(image)
