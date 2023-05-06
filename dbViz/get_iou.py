'''Calculate IOU'''
import torch
import random
from options import options
import os
import argparse
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from model import get_model, get_teacher_model
from data import get_data, make_planeloader

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

def calculate_iou_simple(pred_arr1, pred_arr2):
    diff = pred_arr1.shape[0] - (pred_arr1 - pred_arr2).count_nonzero()
    iou = diff / pred_arr1.shape[0]
    return iou.cpu()


'''
Here we load from a directory that has the following structure: 
args.load_net = /path/to/networks/

/path/to/networks/
    /net1/
        /predictions/
            /net1_preds1.pth
            /net1_preds2.pth
            /net1_preds3.pth

    /net2/
        /predictions/
            /net2_preds1.pth
            /net2_preds2.pth
            /net2_preds3.pth

    .
    .
    .
    
'''
torch.manual_seed(args.set_data_seed)
# LOAD DATASET
trainloader, testloader = get_data(args)
if args.baseset == 'MNIST':
    mnist_trainloader, mnistm_trainloader = trainloader
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []

#Prepare models
if args.baseset == 'MNIST':
    teacher = get_model(args, device)
    ind_student = get_model(args, device)
    dist_student = get_model(args, device)

teacher = get_teacher_model(args, device)
ind_student = get_model(args, device)
dist_student = get_model(args, device)
#print(torch.load(args.load_net, map_location=device).keys())
teacher.load_state_dict(torch.load(args.load_net, map_location=device))
#teacher.load_state_dict(torch.load('dist_models/mnist_mnist_cbg_mnistm_ViT.pt'))
ind_student.load_state_dict(torch.load(args.load_ind_net))
dist_student.load_state_dict(torch.load(args.load_dist_net))
 
#Create Planes
preds = {'Teacher': [], 'Dist': [], 'Ind': []}
iou = {'Teacher-Dist': None, 'Teacher-Ind': None}
count = 0
for (data_mnist, target_mnist), (data_mnistm, target_mnistm) in zip(mnist_trainloader, mnistm_trainloader):
    if count == 40:
        break
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
    preds_teacher = [torch.argmax(a).item() for a in decision_boundary(args, teacher, planeloader, device, args.teacher_net)]
    preds_ind = [torch.argmax(a).item() for a in decision_boundary(args, ind_student, planeloader, device, args.net)]
    preds_dist = [torch.argmax(a).item() for a in decision_boundary(args, dist_student, planeloader, device, args.net)]
    preds['Teacher'].extend(preds_teacher)
    preds['Dist'].extend(preds_dist)
    preds['Ind'].extend(preds_ind)
    count+=1

iou['Teacher-Dist'] = calculate_iou_simple(torch.LongTensor(preds['Teacher']), torch.LongTensor(preds['Dist']))
iou['Teacher-Ind'] = calculate_iou_simple(torch.LongTensor(preds['Teacher']), torch.LongTensor(preds['Ind']))

print(iou)
