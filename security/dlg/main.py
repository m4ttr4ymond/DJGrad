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
                    help='the path to customized image.')
args = parser.parse_args()

def main(index, image=''):
    for tries in range(0, 10):
        EPS = 0.001
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
        # net2 = copy.deepcopy(net)

        net2.apply(weights_init)
        # net2.apply(weights_init2)

        # with torch.no_grad():
        #     for i, layer in enumerate(net2.body):
        #         if i % 2 == 0:
        #             layer.weight = torch.nn.parameter.Parameter(torch.clamp(layer.weight + torch.FloatTensor(layer.weight.shape).uniform_(-EPS, EPS).to(device), min=-0.5, max=0.5))
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
            # random = torch.rand(gy.shape).to(device)
            # gy = torch.where(random > 0, gy, original_dy_dx2[i])
            gy = gy + original_dy_dx[i]

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
                for gx, gy in zip(dummy_dy_dx, original_dy_dx2): 
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

            plt.savefig('outputs/output{}.png'.format(index))
            return


all_classes = random.sample(range(50000), 1000)
for ind in all_classes:
    main(ind)
