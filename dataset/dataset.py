from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as et
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
import yaml

from dataset.augmentations import transforms


class DetectionDataset(Dataset):
    def __init__(self, dir_path, classes, transforms=None, cell=8):
        self.transforms = transforms
        self.dir_path = dir_path
        self.classes = classes
        self.cell = cell
        
        # Получение путей всех изображений в отсортированном порядке
        self.image_paths = glob.glob(f"{self.dir_path}/images/*.jpeg")

        # Получение только названий файлов в отсортированном порядке
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)


    def __len__(self):
        return len(self.all_images)


    def get_image(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, "images", image_name)
        image = np.array(Image.open(image_path))
        return image 


    def get_bboxes(self, idx, image):
        image_name = self.all_images[idx]
        image_width = image.shape[1]
        image_height = image.shape[0]

        annot_filename = image_name[:-5] + '.txt'
        annot_file_path = os.path.join(self.dir_path, 'labels', annot_filename)

        with open(annot_file_path) as fin:
            bboxes = {'bboxes': [], 'labels': []}
            
            # Пропускаем первую строку, так как она содержит количество меток
            num_labels = int(fin.readline().strip())
            
            # Читаем координаты ограничивающих рамок
            for line in fin:
                line = line.strip().split()
                if len(line) == 4:
                    x_min, y_min, x_max, y_max = map(int, line)
                    
                    # Преобразуем координаты в формат [x_min, y_min, x_max, y_max]
                    cords = np.array([x_min, y_min, x_max, y_max])
                    
                    # Преобразуем координаты в доли от ширины и высоты изображения
                    cords[[0, 2]] = cords[[0, 2]] / image_width
                    cords[[1, 3]] = cords[[1, 3]] / image_height
                    
                    # Убеждаемся, что координаты находятся в пределах [0, 1]
                    cords = np.clip(cords, 0, 1)
                    
                    # Добавляем ограничивающие рамки и метки в список
                    bboxes['bboxes'].append(cords)
                    bboxes['labels'].append(0)  # Здесь можно указать правильный label при необходимости

        return bboxes


    def apply_transforms(self, image, bboxes):
        if self.transforms:
            sample = self.transforms(image=image, 
                                     bboxes=bboxes['bboxes'], 
                                     labels=bboxes['labels'])
            image = sample['image']
            bboxes = {'bboxes': sample['bboxes'], 'labels': sample['labels']}
        return image, bboxes
    

    def form_targets(self, image, bboxes):
        image_width = image.shape[2]
        image_height = image.shape[1]
        
        gun_tensor = torch.zeros(5, image_height // self.cell, image_width // self.cell)
        
        for box, label in zip(bboxes['bboxes'], bboxes['labels']):

            b_x = (box[0] + box[2]) / 2 / self.cell
            b_y = (box[1] + box[3]) / 2 / self.cell
            
            b_w = np.log(np.abs(box[2] - box[0]) / self.cell + 1.0e-8)
            b_h = np.log(np.abs(box[3] - box[1]) / self.cell + 1.0e-8)
            
            b_x_index = int(b_x)
            b_x_shift = b_x - b_x_index
            
            b_y_index = int(b_y)
            b_y_shift = b_y - b_y_index 
            
            if self.classes[label] == 'gun':
                gun_tensor[:, b_y_index, b_x_index] = torch.tensor(
                    [1, b_y_shift, b_x_shift, b_h, b_w])
            
        return gun_tensor


    def __getitem__(self, idx):
        image = self.get_image(idx)
        bboxes = self.get_bboxes(idx, image)
        image, bboxes = self.apply_transforms(image, bboxes)
        targets = self.form_targets(image, bboxes)
        return {'img': image, 'targ': targets} 
        

def build_datasets(path, transforms=transforms):
    stream = open(os.path.join(path, "data.yaml"), "r")
    config = yaml.safe_load(stream)

    train_ds = DetectionDataset(
        os.path.join(path, 'train'),
        classes=config['names'],
        transforms=transforms['train']
    )

    test_ds = DetectionDataset(
        os.path.join(path, 'valid'),
        classes=config['names'],
        transforms=transforms['valid']
    )

    return train_ds, test_ds
