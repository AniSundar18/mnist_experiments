'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms
from utils import produce_plot_alt,produce_plot_x,produce_plot_sepleg
import os
import argparse
import time

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time


def colorize_mnist(data):
    bs = data.size(0)
    orig = data.repeat(1, 3, 1, 1)
    rand_color_bg = torch.rand(size=[bs, 3]).cuda()
    rand_color_fg = torch.rand(size=[bs, 3]).cuda()
    color_bg = torch.zeros_like(orig)
    color_fg = torch.zeros_like(orig)
    for i in range(data.size(0)):
        digit_bg = orig[i,0,:,:]>=0.
        digit_fg = orig[i,0,:,:]>0.

        color_bg[i,0,digit_bg] = rand_color_bg[i,0] 
        color_bg[i,1,digit_bg] = rand_color_bg[i,1] 
        color_bg[i,2,digit_bg] = rand_color_bg[i,2]
        
        color_fg[i,0,digit_fg] = rand_color_fg[i,0] 
        color_fg[i,1,digit_fg] = rand_color_fg[i,1] 
        color_fg[i,2,digit_fg] = rand_color_fg[i,2]
    color_bg = color_bg * (1-orig)
    color_bg = color_bg + (orig)
    color_fg = color_fg * orig
    return orig, color_fg, color_bg

args = options().parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = args.save_net
if args.active_log:
    import wandb
    idt = '_'.join(list(map(str,args.imgs)))
    wandb.init(project="decision_boundaries", name = '_'.join([args.net,args.train_mode,idt,'seed'+str(args.set_seed)]) )
    wandb.config.update(args)

# Data/other training stuff
torch.manual_seed(args.set_data_seed)


trainloader, testloader = get_data(args)

    
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
if args.baseset == 'MNIST':
    mnist_trainloader, mnistm_trainloader = trainloader
    teacher = get_model(args, device)
    ind_student = get_model(args, device)
    dist_student = get_model(args, device)
else:
    net = get_model(args, device)
    test_acc, predicted = test(args, net, testloader, device, 0)
    print("scratch prediction ", test_acc)

    criterion = get_loss_function(args)
    if args.opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = get_scheduler(args, optimizer)

    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


# Train or load base network
print("Training the network or loading the network")

start = time.time()
best_acc = 0  # best test accuracy
best_epoch = 0



if args.load_ind_net is not None:
    teacher.load_state_dict(torch.load(args.load_net))
    ind_student.load_state_dict(torch.load(args.load_ind_net))
    dist_student.load_state_dict(torch.load(args.load_dist_net))
else :
    net.load_state_dict(torch.load(args.load_net))
    



end = time.time()
simple_lapsed_time("Time taken to train/load the model", end-start)
if args.baseset == 'MNIST':
    data_mnist, target_mnist = next(iter(mnist_trainloader)) 
    data_mnistm, target_mnistm = next(iter(mnistm_trainloader))
    mnist_orig, target_mnist = data_mnist.to(device), target_mnist.to(device)
    mnistm, target_mnistm = data_mnistm.to(device), target_mnistm.to(device)    
    mnist, mnist_cfg, mnist_cbg = colorize_mnist(mnist_orig)
    data_dict = {'mnist':mnist, 'mnist_cfg': mnist_cfg, 'mnist_cbg':mnist_cbg, 'mnistm':mnistm}
    target_dict = {'mnist': target_mnist, 'mnist_cfg': target_mnist, 'mnist_cbg': target_mnist, 'mnistm':target_mnistm}
    images = []
    labels = []
     
    if 'mnist' in  args.plane_datasets:
        idxs = random.sample(range(1, len(data_dict['mnist'])), 3)
        for idx in idxs:
            images.append(data_dict['mnist'][idx])
            labels.append(target_dict['mnist'][idx])
    if 'mnist_cbg' in  args.plane_datasets:
        idxs = random.sample(range(1, len(data_dict['mnist_cbg'])), 3)
        for idx in idxs:
            images.append(data_dict['mnist_cbg'][idx])
            labels.append(target_dict['mnist_cbg'][idx])
    if 'mnistm' in  args.plane_datasets:
        idxs = random.sample(range(1, len(data_dict['mnistm'])), 3)
        for idx in idxs:
            images.append(data_dict['mnistm'][idx])
            labels.append(target_dict['mnistm'][idx])
    planeloader = make_planeloader(images, args)
    preds_teacher = decision_boundary(args, teacher, planeloader, device)
    preds_ind = decision_boundary(args, ind_student, planeloader, device)
    preds_dist = decision_boundary(args, dist_student, planeloader, device)
    net_names = ['teacher', 'ind_student', 'dist_student']
    preds = [preds_teacher, preds_ind, preds_dist]
    for net_name, pred in zip(net_names, preds):
        produce_plot_sepleg(net_name, pred, planeloader, images, labels, trainloader, title = 'best', temp=1.0,true_labels = None)

