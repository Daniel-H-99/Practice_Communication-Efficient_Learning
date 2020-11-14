#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist
from models.test import test_img
    
def main(iid=False):
    # parse args
    args = args_parser()
    args.iid=iid
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    TAG = "{}_C_{}_iid_{}".format(args.model, args.frac, args.iid)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users_train = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    net_best = None
    best_loss = None
    last_round = 0
    for iter in range(1, args.epochs + 1):
        totp = 0 
        w_glob = None
        loss_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            w_local, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            loss_locals.append(copy.deepcopy(loss))

            p = len(dict_users_train[idx]) / len(dataset_train)
            totp += p

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
                for k in w_glob.keys():
                    w_glob[k] *= p
            else:
                for k in w_glob.keys():
                    w_glob[k] += p * w_local[k]

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], totp)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        
        # update checkpoint
        last_round = iter
        print('[{}] Round {:3d}, Average loss {:.3f}'.format(TAG, last_round, loss_avg))
        torch.save(w_glob, './ckpt/{}_round{}.pt'.format(TAG, last_round))
        loss_train.append(loss_avg)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        if acc_test >= 99:
            break
            
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/{}.png'.format(TAG))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

if __name__ == '__main__':
    main(iid=True)