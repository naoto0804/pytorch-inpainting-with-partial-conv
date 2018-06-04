import glob
import random
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, img_transform, mask_transform, split='train'):
        super(Dataset, self).__init__()
        self.root = root
        self.img_root = '{:s}/{:s}'.format(self.root, 'images')
        self.mask_root = '{:s}/{:s}'.format(self.root, 'masks')
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        with open('{:s}/{:s}.txt'.format(self.root, split)) as f:
            self.paths = ['{:s}/{:s}'.format(self.img_root, p.strip()) for p in
                          f.readlines()]

        self.mask_paths = glob.glob('{:s}/*'.format(self.mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
