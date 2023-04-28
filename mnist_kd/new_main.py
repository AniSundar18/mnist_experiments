from __future__ import print_function
import argparse
from vit_pytorch import ViT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import sys
import matplotlib.pyplot as plt
from mnist_m import MNISTM
from distiller_zoo import DistillKL, HintLoss
from crd.criterion import CRDLoss
import random
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = {}
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out['hint'] = x
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        out['acts'] = x
        #x = self.dropout2(x)
        out['crd'] = x
        x = self.fc2(x)
        out['logits'] = x
        #output = F.log_softmax(x, dim=1)
        return out

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





def train(args, model, model_t, device, dataloaders, optimizer, epoch, criterion_cls, criterion_kd):
    mnist_train_loader = dataloaders[0]
    mnistm_train_loader = dataloaders[1]
    model.train()
    unique_activations = None
    batch_idx = 0
    lossF = None
    for (data_mnist, target_mnist), (data_mnistm, target_mnistm) in zip(mnist_train_loader, mnistm_train_loader):
        mnist_orig, target_mnist = data_mnist.to(device), target_mnist.to(device)
        mnistm, target_mnistm = data_mnistm.to(device), target_mnistm.to(device)    
        if args.model != 'FCNet':
            mnist, mnist_cfg, mnist_cbg = colorize_mnist(mnist_orig)
            data_dict = {'mnist':mnist, 'mnist_cfg': mnist_cfg, 'mnist_cbg':mnist_cbg, 'mnistm':mnistm}
            target_dict = {'mnist': target_mnist, 'mnist_cfg': target_mnist, 'mnist_cbg': target_mnist, 'mnistm':target_mnistm}
        elif args.model == 'FCNet':
            mnist, mnist_cfg, mnist_cbg = colorize_mnist(mnist_orig[:,:784].reshape(mnist_orig.shape[0],1, 28, 28))
            mnist = mnist_orig
            mnist_cbg = torch.flatten(mnist_cbg, start_dim  = 1, end_dim = -1)
            mnist_cfg = torch.flatten(mnist_cfg, start_dim  = 1, end_dim = -1)
            data_dict = {'mnist':mnist, 'mnist_cfg': mnist_cfg, 'mnist_cbg':mnist_cbg, 'mnistm':mnistm}
            target_dict = {'mnist': target_mnist, 'mnist_cfg': target_mnist, 'mnist_cbg': target_mnist, 'mnistm':target_mnistm}
        for dt in range(len(args.train_datasets)):
            dtype = args.train_datasets[dt]
            if dt==0:
                tot_data = data_dict[dtype]
                tot_target = target_dict[dtype]
            else:
                tot_data = torch.cat([tot_data, data_dict[dtype]], dim=0)
                tot_target = torch.cat([tot_target, target_dict[dtype]], dim=0)
        tot_size = tot_data.size(0)
        optimizer.zero_grad()
        output = model(tot_data)
        if args.model == 'ViT':
            cls_loss = criterion_cls(output, tot_target)
        else:
            cls_loss = criterion_cls(output['logits'], tot_target)
        if args.distill is None:
            tot_loss = cls_loss
        else:
            output_t = model_t(tot_data)
            if args.distill == 'kd':
                if args.teacher_model == 'ViT':
                    logits_t = output_t
                else:
                    logits_t = output_t['logits']
                if args.model == 'ViT':
                    logits_s = output
                else:
                    logits_s = output['logits']
                kd_loss = criterion_kd(logits_s, logits_t)
                tot_loss = kd_loss
                
            elif args.distill == 'hint':
                kd_loss = criterion_kd(output['hint'], output_t['hint'])
                tot_loss = cls_loss + (2*kd_loss)
            #elif args.distill == 'crd':

        tot_loss.backward()
        optimizer.step()
        lossF = tot_loss.item()
        if batch_idx % args.log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        batch_idx += 1
    return lossF

def test(args, model, device, test_loader, epoch):
    model.eval()
    #test_loss = 0
    correct_orig, correct_fg, correct_bg, correct_m = 0, 0, 0, 0
    with torch.no_grad():
        for (data, target), (data_mnistm, target_mnistm) in zip(test_loader[0], test_loader[1]):
            #data, target = data.to(device), target.to(device)
            #data_mnistm, target_mnistm = data_mnistm.to(device), target_mnistm.to(device)
            data, target = data.to(device), target.to(device)
            data_mnistm, target_mnistm = data_mnistm.to(device), target_mnistm.to(device)

            data_orig, data_fg, data_bg = colorize_mnist(data)

            if 'mnist' in args.test_datasets:
                output = model(data_orig)
                if args.model == 'ViT':
                    pred = output.argmax(dim=1, keepdim=True)
                else :
                    pred = output['logits'].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_orig += pred.eq(target.view_as(pred)).sum().item()

            if 'mnist_cfg' in args.test_datasets:
                output = model(data_fg)
                if args.model == 'ViT':
                    pred = output.argmax(dim=1, keepdim=True)  
                else :
                    pred = output['logits'].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_fg += pred.eq(target.view_as(pred)).sum().item()

            if 'mnist_cbg' in args.test_datasets:
                output = model(data_bg)
                if args.model == 'ViT':
                    pred = output.argmax(dim=1, keepdim=True)  
                else :
                    pred = output['logits'].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_bg += pred.eq(target.view_as(pred)).sum().item()

            if 'mnistm' in args.test_datasets:
                output = model(data_mnistm)
                if args.model == 'ViT':
                    pred = output.argmax(dim=1, keepdim=True)  
                else :
                    pred = output['logits'].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_m += pred.eq(target_mnistm.view_as(pred)).sum().item()
    print ('\n')
    print ("Epoch %d" %(epoch))
    if args.distill is not None:
        print (args.distill)
    if 'mnist' in args.test_datasets:
        print ('Accuracy MNIST: %f' %(100. * correct_orig/len(test_loader[0].dataset)))
    if 'mnist_cfg' in args.test_datasets:
        print ('Accuracy MNIST Color FG: %f' %(100. * correct_fg/len(test_loader[0].dataset)))
    if 'mnist_cbg' in args.test_datasets:
        print ('Accuracy MNIST Color BG: %f' %(100. * correct_bg/len(test_loader[0].dataset)))
    if 'mnistm' in args.test_datasets:
        print ('Accuracy MNIST_M: %f' %(100. * correct_m/len(test_loader[1].dataset)))





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train_datasets', nargs="+")
    parser.add_argument('--test_datasets', nargs="+")
    parser.add_argument('--same_init', type=bool, default=False)
    parser.add_argument('--usage', type=str, default='train')
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--teacher_model', type=str, default=None)
    parser.add_argument('--distill', type=str, default=None)
    parser.add_argument('--SEED', type=int, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--plot_curves', action='store_true', default=False,
                        help='For Plotting Loss vs Epoch curves')
    parser.add_argument('--train-few', action='store_true', default=False,
                        help='Option to use if being trained only on fewer training images')
    
    # CRD parameters
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    args = parser.parse_args()
    seed = args.SEED
    LOSSES = []
    torch.manual_seed(seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    transform_m=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])

    mnistm_train = MNISTM('./data', train=True, download=True, transform=transform_m)
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)

    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    if args.SEED is not None:
        mnist_train_loader = torch.utils.data.DataLoader(mnist_train,**train_kwargs, generator=torch.Generator().manual_seed(seed))
    else :
        mnist_train_loader = torch.utils.data.DataLoader(mnist_train,**train_kwargs)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, **test_kwargs)
     
    mnistm_test = MNISTM('./data', train=False, transform=transform_m)
    if args.SEED is not None:
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_train,**train_kwargs, generator=torch.Generator().manual_seed(seed))
    else :
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_train,**train_kwargs)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_test, **test_kwargs)
    data_t = None
    logits_t = None
    if args.model == 'CNN':
        model = Net().to(device)
    elif args.model == 'ViT':
        model = ViT(image_size = 28,
                    patch_size = 7,
                    num_classes = 10,
                    dim = 512,
                    depth = 3,
                    heads = 8,
                    mlp_dim = 512,
                    dropout = 0.2,
                    emb_dropout = 0.1
                    ).to(device)
    print("Training On: ", args.train_datasets) 
    if args.usage == 'train':
        criterion_cls = nn.CrossEntropyLoss()
        criterion_kd = DistillKL(8)
        model_t = None
        trainable_list = nn.ModuleList([])
        trainable_list.append(model)
        if args.distill is not None or args.freeze_layer:
            if args.teacher_model == 'CNN':
                model_t = Net().to(device)
            elif args.teacher_model == 'ViT':
                model_t = ViT(image_size = 28,
                            patch_size = 7,
                            num_classes = 10,
                            dim = 512,
                            depth = 3,
                            heads = 8,
                            mlp_dim = 512,
                            dropout = 0.2,
                            emb_dropout = 0.1
                            ).to(device)

            path_t = './models/%s.pt' %(args.pretrained_model) 
            pretrained_dict = torch.load(path_t)
            model_t.load_state_dict(pretrained_dict)
            model_t.eval()

        if args.distill == 'kd':
            criterion_kd = DistillKL(8)
        elif args.distill == 'hint':
            criterion_kd = HintLoss()
        elif args.distill == 'crd':
            args.s_dim = 128
            args.t_dim = 128
            args.n_data = len(mnist_train_loader)*args.batch_size    
            criterion_kd = CRDLoss(args)
            trainable_list.append(criterion_kd.embed_s)
            trainable_list.append(criterion_kd.embed_t)


        optimizer = optim.Adadelta(trainable_list.parameters(), lr=args.lr)


        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            LOSSES.append(train(args, model, model_t, device, [mnist_train_loader, mnistm_train_loader], optimizer, epoch, criterion_cls, criterion_kd))
            test(args, model, device, [mnist_test_loader, mnistm_test_loader], epoch)
            scheduler.step()

        if args.save_model:
            key = ''
            for dt in args.train_datasets:
                key = key + '%s_'%(dt)
            key = key[:-1]
            if args.teacher_model is not None:
                key = key + '_' + args.model + '_from_' + args.teacher_model
            else:
                key = key + '_' + args.model 
            if args.distill:
                key = key + '_' + args.distill
            torch.save(model.state_dict(), "./models/%s.pt" %(key))
            if args.plot_curves:
                EPOCHS = list(range(1, args.epochs + 1))
                plt.scatter(EPOCHS, LOSSES)
                plt.plot(EPOCHS, LOSSES)
                plt.title('Training Loss v Epochs: ' + key)
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.savefig("./models/%s.png" %(key))
    else:
        assert args.pretrained_model
        path = './models/%s.pt' %(args.pretrained_model)
        pretrained_dict = torch.load(path)
        model.load_state_dict(pretrained_dict)
        test(args, model, device, [mnist_test_loader, mnistm_test_loader], 0)


if __name__ == '__main__':
    main()
