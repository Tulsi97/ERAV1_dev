import albumentations as A
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2

class CustomResnetTransforms:
    def train_transforms(means, stds):
        return A.Compose(
                [
                    A.Normalize(mean=means, std=stds, always_apply=True),
                    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                    A.RandomCrop(height=32, width=32, always_apply=True),
                    A.HorizontalFlip(),
                    A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0, p=1.0),
                    ToTensorV2(),
                ]
            )
    
    def test_transforms(means, stds):
         return A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )