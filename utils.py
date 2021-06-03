import os
import numpy as np
import torch
import torchvision.transforms as T
import cv2
import albumentations as A
from PIL import Image
from torchvision.transforms.transforms import CenterCrop, ToPILImage

__all__ = ['GetPathNLabel', 'ToTensor', 'ToTensor', 'MaskDataset']


class GetPathNLabel:
    def __init__(self, splitted_folders):
        self._train_image_dir = '/opt/ml/input/data/train/images'
        self.splitted_folders = splitted_folders

    def get_image(self, train_image_dir):
        '''
        Args:
            train_image_dir : parent dir of image dataset
        Return:
            train_imgs : child dir of img dataset with img name
        '''
        # train_img_folders = os.listdir(train_image_dir)
        # train_img_folders = [
        #     folder for folder in train_img_folders if folder[:2] != '._']
        train_img_folders = self.splitted_folders
        train_imgs = []
        for path in train_img_folders:
            imgs = os.listdir(os.path.join(train_image_dir, path.strip('._')))
            imgs = [os.path.join(path, img) for img in imgs if img[:2] != '._']
            train_imgs += imgs
        return train_imgs

        # for i in range(len(self.splitted_folders)):
        #     self.splitted_folders[i] = self._train_image_dir + \
        #         self.splitted_folders[i]
        # return self.splitted_folders

    def make_label(self, train_imgs):
        '''
        Args:
            train_imgs : child dir of img dataset with img name
        Return:
            labels : labels of image dataset
        '''
        labels = []
        train_imgs_splitted = [train_img.split(
            '/') for train_img in train_imgs]
        # print(train_imgs_splitted)
        for img in train_imgs_splitted:
            meta = img[0].split('_')
            image = img[1].split('.')
            mask = 12 if image[0] == 'normal' else 6 if image[0] == 'incorrect_mask' else 0
            sex = 0 if meta[1] == 'male' else 3
            age = 0 if int(meta[3]) < 30 else 1 if int(
                meta[3]) >= 30 and int(meta[3]) < 58 else 2  # apply age threshold -> age > 58 -> class 2
            labels.append(mask + sex + age)
        return labels

    def call(self):
        '''
        Return:
            img path with label
        '''
        train_imgs = self.get_image(self._train_image_dir)
        labels = self.make_label(train_imgs)
        train_imgs = [os.path.join(self._train_image_dir, train_img)
                      for train_img in train_imgs]
        return train_imgs, labels


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))  # (C, H, W)
        return {'image': torch.FloatTensor(image),
                'label': torch.LongTensor(label)}


to_tensor = T.Compose([
                      ToTensor()
                      ])


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, labels, transforms=to_tensor, augmentations=None):
        self.img_path = img_path
        self.labels = labels

        self.transforms = transforms
        self.augmentations = augmentations
        self.normalization = T.Compose([
            T.ToPILImage(),
            # T.Resize((256, 256), Image.BILINEAR),
            T.CenterCrop((300, 256)),
            T.ToTensor(),
            T.Normalize(mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245))
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = cv2.imread(self.img_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        ycrcb_planes = cv2.split(img_ycrcb)
        # 밝기 성분에 대해서만 histogram equalization
        ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        dst_ycrcb = cv2.merge(ycrcb_planes)
        dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2RGB)
        dst = (dst/255.).astype('float32')

        label = np.array(self.labels[index]).astype(float)

        data = {'image': dst, 'label': label}

        if self.augmentations:
            data['image'] = self.augmentations(image=data['image'])['image']

        if self.transforms:
            data = self.transforms(data)
            data['image'] = self.normalization(data['image'])

        return data
