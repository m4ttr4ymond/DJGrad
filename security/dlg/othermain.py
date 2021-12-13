# -*- coding: utf-8 -*-
import argparse
import os
import copy
import random
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--gpu', type=str,default="0",
                    help='GPU')
parser.add_argument('--eps', type=float,default=0.01,
                    help='Epsilon')
parser.add_argument('--batch', type=int,default=0,
                    help='Batch')
args = parser.parse_args()

def main(index, image=''):
    for tries in range(0, 10):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print("Running on %s" % device)

        dst = datasets.CIFAR100("~/.torch", download=True)
        tp = transforms.ToTensor()
        tt = transforms.ToPILImage()

        img_index = index
        gt_data = tp(dst[img_index][0]).to(device)
        gt_data2 = tp(dst[img_index + 1][0]).to(device)

        if len(image) > 1:
            gt_data = Image.open(image)
            gt_data = tp(gt_data).to(device)

        gt_data = gt_data.view(1, *gt_data.size())
        gt_data2 = gt_data2.view(1, *gt_data2.size())
        gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
        gt_label2 = torch.Tensor([dst[img_index + 1][1]]).long().to(device)
        gt_label2 = gt_label2.view(1, )
        gt_onehot_label = label_to_onehot(gt_label)
        gt_onehot_label2 = label_to_onehot(gt_label2)


        plt.imshow(tt(gt_data[0].cpu()))

        from models.vision import LeNet, weights_init, weights_init2, weights_init_copy
        net = LeNet().to(device)
        net2 = LeNet().to(device)

        torch.manual_seed(1234)

        net.apply(weights_init)
        net2 = copy.deepcopy(net)

        # net2.apply(weights_init)
        # net2.apply(weights_init2)

        with torch.no_grad():
            for i, layer in enumerate(net2.body):
                if i % 2 == 0:
                    layer.weight = torch.nn.parameter.Parameter(torch.clamp(layer.weight + torch.FloatTensor(layer.weight.shape).uniform_(-args.eps, args.eps).to(device), min=-0.5, max=0.5))
        #             # layer.weight = torch.nn.parameter.Parameter(torch.clamp(layer.weight, min=-0.5, max=0.5))
        #             # layer.weight = torch.nn.parameter.Parameter(layer.weight)

        criterion = cross_entropy_for_onehot

        # compute original gradient 
        pred = net(gt_data)
        pred2 = net2(gt_data)

        y = criterion(pred, gt_onehot_label)
        y2 = criterion(pred2, gt_onehot_label)

        dy_dx = torch.autograd.grad(y, net.parameters())
        dy_dx2 = torch.autograd.grad(y2, net2.parameters())

        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        original_dy_dx2 = list((_.detach().clone() for _ in dy_dx2))

        # for i, gy in enumerate(original_dy_dx):
        #     random = torch.rand(gy.cpu().numpy().shape).to(device)
        #     gy = gy + random

        # for i, gy in enumerate(original_dy_dx):
        #     # random = torch.rand(gy.shape).to(device)
        #     # gy = torch.where(random > 0, gy, original_dy_dx2[i])
        #     gy = gy + original_dy_dx2[i]

        for i, gy in enumerate(original_dy_dx2):
            random = torch.rand(gy.shape).to(device)
            gy = torch.where(random > 0, gy, original_dy_dx2[i])
            original_dy_dx2[i] = gy + original_dy_dx[i]

        # generate dummy data and label
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

        plt.imshow(tt(dummy_data[0].cpu()))

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        best_loss = 1000000
        history = []
        for iters in range(300):
            def closure():
                optimizer.zero_grad()

                dummy_pred = net2(dummy_data) 
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net2.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                
                return grad_diff
            
            optimizer.step(closure)
            if iters % 10 == 0: 
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())
                history.append(tt(dummy_data[0].cpu()))

            if current_loss < best_loss:
                best_loss = current_loss

        if best_loss < 0.1:
            plt.figure(figsize=(12, 8))
            for i in range(30):
                plt.subplot(3, 10, i + 1)
                plt.imshow(history[i])
                plt.title("iter=%d" % (i * 10))
                plt.axis('off')

            plt.savefig('outputs-{}/output{}.png'.format(args.eps, index))
            return


all_classes = [[12580, 39473, 11261, 31684, 38457, 27046, 38376, 24029, 29708, 11933,
               38821, 18981, 46146, 30492, 33904, 36736, 6203,  3018,  6962,  25654,
               6349,  23502, 37978, 16033, 23446, 4361,  2271,  30534, 13721, 22000,
               13440, 44422, 15235, 16693, 46452, 34212, 24283, 5943,  100,   1140,
               38612, 20897, 2979,  23738, 37007, 11394, 20622, 37444, 16317, 20005],
               [17294, 15808, 14021, 23200, 42735, 46235, 4884,  47219, 9981,  31171,
               36199, 10431, 47828, 30746, 49665, 1180,  22472, 22583, 41305, 610,
               22360, 3754,  43603, 46140, 26728, 34693, 49458, 12909, 22442, 49752,
               23631, 35869, 49260, 16029, 19875, 27860, 23741, 33469, 36002, 20249,
               33855, 5992,  47789, 34817, 41417, 32972, 45765, 33358, 47246, 15836],
               [45286, 20036, 23763, 9382,  46339, 47946, 48131, 44042, 48708, 29984,
               41537, 47481, 34375, 20571, 47425, 34200, 18095, 49850, 48229, 44196,
               28761, 19153, 18883, 1142,  11024, 38441, 7637,  21867, 8157,  23769,
               42136, 8600,  20294, 10532, 15553, 34946, 41186, 46012, 42969, 8216,
               2516,  22189, 17024, 22529, 2538,  36873, 2294,  49295, 49910, 33127],
               [17391, 19608, 19225, 21248, 42869, 12946, 11741, 47424, 36897, 44885,
               36666, 43263, 49215, 39469, 35530, 5623,  49630, 18394, 13770, 47262,
               31346, 23113, 35142, 30709, 17533, 49840, 21299, 28986, 12463, 39341,
               45577, 49633, 8098,  851,   45659, 22101, 46674, 16984, 48656, 27354,
               42967, 10043, 32039, 15757, 48512, 37004, 8944,  28779, 18829, 45238]]

for ind in all_classes[args.batch]:
    main(ind)
