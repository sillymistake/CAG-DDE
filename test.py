import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import argparse
import glob
import numpy as np
from tqdm import tqdm
import os
import random
from PIL import Image
from tensorboardX import SummaryWriter

from dataloader import MyDataset_TR, MyDataset_TE
from model.ours_pair import MyNet
from metric import cal_fm, cal_mae, cal_sm, cal_em
from load_data import *

mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([3,1,1])).float()
norm = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([3,1,1])).float()

parser = argparse.ArgumentParser(description='Training with Pytorch')
parser.add_argument('--dir_name', type=str, default='test',
                    help='Directory for storing checkpoint models')
parser.add_argument('--img_size', type=tuple, default=(448,448),
                    help='size of input image (w,h)')
parser.add_argument('--is_origin', type=bool, default=False,
                    help='input original image')
parser.add_argument('--batch_size_test', type=int, default=4,
                    help='testing batch size')
parser.add_argument('--load_model', type=str, default='./save_model/best.pth',
                    help='name for pretrained model')
parser.add_argument('--dataset', type=str, choices=['davis', 'davsod', 'fbms', 'segv2'], default='davis',
                    help='dataset for training and testing')
parser.add_argument('--is_save', type=bool, default=False,
                    help='save output')


args = parser.parse_args()
print(args)

save_dir = './result/'
save_dir = save_dir + args.dir_name
try:
    os.makedirs(save_dir)
except:
    pass

# Function
def load_pretrained_model(net, model_path):
    print('Load pretrained ', model_path, '...')
    pretrain_dict = torch.load(model_path)
    model_dict = {}
    state_dict = net.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
        else:
            print('miss', k)
    state_dict.update(model_dict)
    net.load_state_dict(state_dict)
    return net

def eval_model(net, tes_dataloader, test_num):
    Fmeasure = cal_fm(test_num)
    MAE = cal_mae()
    Sm = cal_sm()
    net.eval()
    print('-----Testing-----')
    test_iter = tqdm(tes_dataloader, ncols=150)
    for sample in test_iter:
        image = sample['img']
        flow = sample['flo']
        label = sample['lbl']
        name = sample['name']

        preds = net(image.float().cuda(),flow.float().cuda())
        for i, pred in enumerate(preds['final']):
            pred = torch.sigmoid(preds['final'][i]).squeeze().cpu().data.numpy()
            shape = label[i].shape[::-1] # hw -> wh
            # save image
            if args.is_save == True:
                if args.dataset == 'davis':
                    my_dir = save_dir + '/OURS_DAVIS/' + name[i].split('/')[-2]
                elif args.dataset == 'davsod':
                    my_dir = save_dir + '/OURS_DAVSOD/' + name[i].split('/')[-3]
                elif args.dataset == 'fbms':
                    my_dir = save_dir + '/OURS_FBMS/' + name[i].split('/')[-2]
                elif args.dataset == 'segv2':
                    my_dir = save_dir + '/OURS_SEGV2/' + name[i].split('/')[-3]
                else:
                    print('error!')
                    return
                try:
                    os.makedirs(my_dir)
                except:
                    pass

                img_name = name[i].split('/')[-1]
                save_img = Image.fromarray(pred*255).convert('L')
                save_img = save_img.resize(shape, resample=Image.BILINEAR)
                save_img.save(my_dir+'/'+img_name)

            pred = np.array(Image.fromarray(pred).resize((shape), resample=Image.BILINEAR))
            Fmeasure.update(pred, np.array(label[i]))
            MAE.update(pred, np.array(label[i]))
            Sm.update(pred, np.array(label[i]))

    fmeasure, maxF, meanF, precision, recall = Fmeasure.show()
    mae = MAE.show()
    sm = Sm.show()

    return maxF, sm, mae

# dataset
if args.dataset == 'davis':
    tra_list, tes_list = load_davis()
elif args.dataset == 'davsod':
    tra_list, tes_list = load_davsod()
elif args.dataset == 'fbms':
    tes_list = load_fbms()
elif args.dataset == 'segv2':
    tes_list = load_segv2()

test_num = len(tes_list)

print('test images:', len(tes_list))


transform_tes = {'size':args.img_size}
tes_dataset = MyDataset_TE(tes_list, is_origin=args.is_origin, transform=transform_tes)
tes_dataloader = DataLoader(tes_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)



# model
net = MyNet(1)
# load pre-trained weights
net = load_pretrained_model(net, args.load_model)

if torch.cuda.is_available():
    net.cuda()

maxF, sm, mae = eval_model(net, tes_dataloader, test_num)

print('maxF: %.4f' % maxF)
print('sm: %.4f' % sm)
print('mae: %.4f' % mae)

# write log
argsDict = args.__dict__
with open(save_dir+'/result_'+args.dataset+'.txt', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------' + '\n')
    f.writelines('maxF: ' + str(maxF) + '\n')
    f.writelines('sm: ' + str(sm) + '\n')
    f.writelines('mae: ' + str(mae) + '\n')