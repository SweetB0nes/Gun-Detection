import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2

transforms = {
    'train': A.Compose([
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)),  # Добавляем значение value для заполнения черным цветом
            A.RandomCrop(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.HueSaturationValue(p=0.2),
            A.RandomGamma(p=0.2),
            A.Normalize(p=1.0),
            ToTensorV2(p=1.0)],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3
        )
    ), 
    'valid': A.Compose([
        A.SmallestMaxSize(max_size=512, always_apply=True),
        A.PadIfNeeded(
            min_height=None, pad_height_divisor=32,
            min_width=None, pad_width_divisor=32,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)),  # Добавляем значение value для заполнения черным цветом
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))
}
