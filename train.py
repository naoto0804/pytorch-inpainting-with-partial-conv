import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from dataset import Dataset
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from util.io import load_ckpt
from util.io import save_ckpt


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        # i = 0
        i = self.num_samples - 1
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--resume', type=str)
args = parser.parse_args()

lambda_dict = {'valid': 1.0, 'hole': 5.0, 'prc': 0.05, 'style': 10.0, 'tv': 0.1}

device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images')
    os.makedirs('{:s}/ckpt')

if not os.path.exists(args.log_dir):
    os.mkdirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

img_transform = transforms.Compose(
    [transforms.Resize(size=(512, 512)), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

mask_transform = transforms.Compose(
    [transforms.Resize(size=(512, 512)), transforms.ToTensor()])

dataset_train = Dataset(args.root, img_transform, mask_transform, 'train')
dataset_val = Dataset(args.root, img_transform, mask_transform, 'val')

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))

model = PConvUNet().to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
else:
    start_iter = 0

for i in tqdm(range(start_iter, args.max_iter)):
    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0
    for key, coef in lambda_dict.items():
        value = coef * loss_dict[key]
        loss += value
        writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.snapshot_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    # TODO visualization

writer.close()
