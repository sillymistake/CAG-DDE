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
from math import exp
from tensorboardX import SummaryWriter

from dataloader import MyDataset_TR, MyDataset_TE
from model.ours_pair import MyNet
from metric import cal_fm, cal_mae, cal_sm, cal_em
from load_data import *
import pytorch_ssim
import pytorch_iou

mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([3,1,1])).float()
norm = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([3,1,1])).float()

parser = argparse.ArgumentParser(description='Training with Pytorch')
parser.add_argument('--dir_name', type=str, default='test',
                    help='Directory for storing checkpoint models')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--epoch_num', type=int, default=100,
                    help='training epoch (default: 100)')
parser.add_argument('--batch_size_train', type=int, default=4,
                    help='training batch size')
parser.add_argument('--ite_size', type=int, default=1,
                    help='batch number for every iteration')
parser.add_argument('--img_size', type=tuple, default=(448,448),
                    help='size of input image (w,h)')
parser.add_argument('--is_origin', type=bool, default=False,
                    help='input original image')
parser.add_argument('--load_model', type=str, choices=['resnet50', 'deeplabv3+'], default='deeplabv3+',
                    help='name for pretrained model')
parser.add_argument('--device_num', type=int, default=1,
                    help='device number for training model')
parser.add_argument('--dataset', type=str, choices=['duts', 'davis', 'davsod', 'mix'], default='mix',
                    help='dataset for training and testing')
parser.add_argument('--lr_sche', type=str, choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'PolynomialLR', 'CosineLR', 'CosineRestart', 'ReduceLROnPlateau'], default='MultiStepLR',
                    help='lr scheduler for training')
parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='adam',
                    help='optimizer for training')


pretrain_model = {'resnet50':'/media/data1/chenpj/.cache/torch/checkpoints/resnet50-19c8e357.pth',
                  'deeplabv3+':'/media/data1/chenpj/vsod/saved_model/duts_pretrained.pth'}

args = parser.parse_args()
print(args)

model_dir = './saved_model/'
model_dir = model_dir + args.dir_name
try:
    os.makedirs(model_dir)
except:
    pass

writer = SummaryWriter(os.path.join(model_dir, 'tensorboard'))

# write log
argsDict = args.__dict__
with open(model_dir+'/args.txt', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')

# Function

# Nx1xHxW
def IoU(pred, label):
    # downsample
    label = F.interpolate(label, pred.size()[2:], mode='bilinear', align_corners=True)

    n,c,h,w = pred.shape
    inter = (pred * label).view(n,-1).sum(dim=1)
    union = (pred + label - (pred * label)).view(n,-1).sum(dim=1)
    iou = inter / (union + 1e-16)

    return iou.unsqueeze(1)

# freeze the bn of the encoder
def freeze_bn(model):
    model.bn1.eval()
    model.bn1.weight.requires_grad = False
    model.bn1.bias.requires_grad = False
    for m in model.layer1.modules():
        if isinstance(m, nn.BatchNorm2d):
            # print(m)
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    for m in model.layer2.modules():
        if isinstance(m, nn.BatchNorm2d):
            # print(m)
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    for m in model.layer3.modules():
        if isinstance(m, nn.BatchNorm2d):
            # print(m)
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    for m in model.layer4.modules():
        if isinstance(m, nn.BatchNorm2d):
            # print(m)
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

# weight init for the decoder
def weight_init(model):
    for m in model.decoder.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                m.bias.data.zero_()


def create_scheduler(scheduler, optimizer):
    if scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # 30
    elif scheduler == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,100], gamma=0.5)
    elif scheduler == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    elif scheduler == 'PolynomialLR':
        lr_lambda = lambda epoch: max((1-float(epoch)/100)**0.9, 0.001)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler == 'CosineLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, last_epoch=-1)
    elif scheduler == 'CosineRestart':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=0, last_epoch=-1)
    elif scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    

def create_optimizer(net, lr, opt_type='adam'):
    if opt_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005) # weight_decay= 0 or 0.0005
    elif opt_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    return optimizer

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
    net.eval()
    print('-----Testing-----')
    test_iter = tqdm(tes_dataloader, ncols=150)
    for sample in test_iter:
        image = sample['img']
        flow = sample['flo']
        label = sample['lbl']

        preds = net(image.float().cuda(),flow.float().cuda())['final']
        for i, pred in enumerate(preds):
            pred = torch.sigmoid(pred).squeeze().cpu().data.numpy()
            shape = label[i].shape[::-1] # hw -> wh
            pred = np.array(Image.fromarray(pred).resize((shape), resample=Image.BILINEAR))
            Fmeasure.update(pred, np.array(label[i]))
            MAE.update(pred, np.array(label[i]))

    fmeasure, maxF, meanF, precision, recall = Fmeasure.show()
    mae = MAE.show()

    return meanF, maxF, mae

# dataset
if args.dataset == 'duts':
    tra_list, tes_list = load_duts()
elif args.dataset == 'davis':
    tra_list, tes_list = load_davis()
elif args.dataset == 'davsod':
    tra_list, tes_list = load_davsod()
elif args.dataset == 'mix':
    tra_list_davis, tes_list_davis = load_davis()
    tra_list_davsod, tes_list_davsod = load_davsod()
    tra_list = tra_list_davis + tra_list_davsod
    # tes_list = tes_list_davsod # select davsod as the main test dataset

# debug
# tra_list = tra_list[:100]
# tes_list = tra_list

train_num = len(tra_list)
test_num_davis = len(tes_list_davis)
test_num_davsod = len(tes_list_davsod)

print('train images:', len(tra_list))
print('test images davis:', len(tes_list_davis))
print('test images davsod:', len(tes_list_davsod))

# transform
transform_tra = {'size':args.img_size, 'flip':True, 'rotate':False, 'jitter':False}
transform_tes = {'size':args.img_size}

tra_dataset = MyDataset_TR(tra_list, is_origin=args.is_origin, transform=transform_tra)
tes_dataset_davis = MyDataset_TE(tes_list_davis, is_origin=args.is_origin, transform=transform_tes)
tes_dataset_davsod = MyDataset_TE(tes_list_davsod, is_origin=args.is_origin, transform=transform_tes)
tra_dataloader = DataLoader(tra_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
tes_dataloader_davis = DataLoader(tes_dataset_davis, batch_size=args.batch_size_train, shuffle=False, num_workers=4, pin_memory=True)
tes_dataloader_davsod = DataLoader(tes_dataset_davsod, batch_size=args.batch_size_train, shuffle=False, num_workers=4, pin_memory=True)


# model
net = MyNet(1)
# load pre-trained weights
net = load_pretrained_model(net, pretrain_model[args.load_model])
freeze_bn(net)
# weight_init(net)


if torch.cuda.is_available():
    if args.device_num != 1:
        net = torch.nn.DataParallel(net, device_ids=range(args.device_num))
        # patch_replication_callback(net)
    net.cuda()

# optimizer
optimizer = create_optimizer(net, args.lr, args.optim)

# lr scheduler
scheduler = create_scheduler(args.lr_sche, optimizer)

# loss
# loss = nn.BCELoss(reduction='mean')
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
mse_loss = nn.L1Loss(reduction='mean')

def fscore_loss(pred, target):
    tp = pred * target
    fs = 1.3 * tp.sum(dim=(1, 2, 3)) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) * 0.3)
    loss = 1 - fs.mean()
    return loss

def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    # fscore_out = fscore_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + iou_out + ssim_out
    return loss


def cal_loss(pred, label, ite_num, t0=10000):
    loss_seg = 0.0
    loss_reg = 0.0

    loss_main = bce_ssim_loss(torch.sigmoid(pred['final']), label)

    label = F.interpolate(label, scale_factor=0.5, mode='nearest')
    loss_seg += bce_loss(torch.sigmoid(pred['layer1'][0]), label) + bce_loss(torch.sigmoid(pred['layer1'][1]), label)
    loss_reg += mse_loss(pred['layer1'][2], IoU(torch.sigmoid(pred['layer1'][0]), label)) + mse_loss(pred['layer1'][3], IoU(torch.sigmoid(pred['layer1'][1]), label))

    label = F.interpolate(label, scale_factor=0.5, mode='nearest')
    loss_seg += bce_loss(torch.sigmoid(pred['layer2'][0]), label) + bce_loss(torch.sigmoid(pred['layer2'][1]), label)
    loss_reg += mse_loss(pred['layer2'][2], IoU(torch.sigmoid(pred['layer2'][0]), label)) + mse_loss(pred['layer2'][3], IoU(torch.sigmoid(pred['layer2'][1]), label))

    label = F.interpolate(label, scale_factor=0.5, mode='nearest')
    loss_seg += bce_loss(torch.sigmoid(pred['layer3'][0]), label) + bce_loss(torch.sigmoid(pred['layer3'][1]), label)
    loss_reg += mse_loss(pred['layer3'][2], IoU(torch.sigmoid(pred['layer3'][0]), label)) + mse_loss(pred['layer3'][3], IoU(torch.sigmoid(pred['layer3'][1]), label))

    label = F.interpolate(label, scale_factor=0.5, mode='nearest')
    loss_seg += bce_loss(torch.sigmoid(pred['layer4'][0]), label) + bce_loss(torch.sigmoid(pred['layer4'][1]), label)
    loss_reg += mse_loss(pred['layer4'][2], IoU(torch.sigmoid(pred['layer4'][0]), label)) + mse_loss(pred['layer4'][3], IoU(torch.sigmoid(pred['layer4'][1]), label))

    loss_reg += mse_loss(pred['layer5'][2], IoU(torch.sigmoid(pred['layer5'][0]), label)) + mse_loss(pred['layer5'][3], IoU(torch.sigmoid(pred['layer5'][1]), label))
    loss_seg += bce_loss(torch.sigmoid(pred['layer5'][0]), label) + bce_loss(torch.sigmoid(pred['layer5'][1]), label)

    # from ITSD [0.1, 0.3, 0.5, 0.7, 0.9, 1.5]
    # loss = 1.5*loss0 + 0.9*loss1 + 0.7*loss2 + 0.5*loss3 + 0.3*loss4 + 0.1*loss5
    # loss = loss_main + max(exp(1-ite_num/t0),1)*loss_seg + min(exp(ite_num/t0-1),1)*loss_reg

    loss = loss_main + loss_seg + loss_reg

    return loss

# train
net.zero_grad()
optimizer.zero_grad()
ite_num = 0
best_mf = 0.0

for epoch in range(1, args.epoch_num+1):
    net.train()
    torch.cuda.empty_cache()
    running_loss = 0.0
    train_loss = 0.0

    train_iter = tqdm(tra_dataloader, ncols=150)
    for i, sample in enumerate(train_iter, start=1):
        image = sample['img']
        flow = sample['flo']
        label = sample['lbl']

        # multi-scale train
        scales = [0.75, 1, 1.25]
        scale = np.random.choice(scales, 1)
        scale_size = (int(args.img_size[1]*scale[0]), int(args.img_size[0]*scale[0])) # (h,w)

        image = F.interpolate(image, size=scale_size, mode='bilinear', align_corners=True).float().cuda()
        flow = F.interpolate(flow, size=scale_size, mode='bilinear', align_corners=True).float().cuda()
        label = F.interpolate(label, size=scale_size, mode='nearest').float().cuda()

        pred = net(image,flow)
        tra_loss = cal_loss(pred, label, ite_num)
        tra_loss.backward()

        running_loss += tra_loss.data

        ite_num += 1

        if i % args.ite_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        # output
        train_loss = running_loss/i
        train_iter.set_description("[Epoch: %d/%d][Iteration: %d] lr:%.4g loss: %.4f" % (epoch, args.epoch_num, ite_num, optimizer.param_groups[0]['lr'], train_loss))

        if ite_num % 200 == 0:
            s_i = min(random.randint(0, args.batch_size_train-1), image.shape[0]-1)
            img_show = image[s_i] * norm.cuda() + mean.cuda()
            flo_show = flow[s_i] * norm.cuda() + mean.cuda()
            lbl_show = label[s_i]
            prd_show = torch.sigmoid(pred['final'][s_i])
            _, h, w = lbl_show.shape
            writer.add_image('Image', torchvision.utils.make_grid([img_show,flo_show,lbl_show.expand(3,h,w),prd_show.expand(3,h,w)], padding=5, pad_value=1), ite_num)
            writer.file_writer.flush()
            del img_show, flo_show, lbl_show, prd_show #, prd_show_x, prd_show_y

    # writer
    writer.add_scalar('Train/loss', train_loss, ite_num)
    writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], ite_num)

    if args.lr_sche == 'ReduceLROnPlateau':
        scheduler.step(train_loss)
    else:
        scheduler.step()

    # save model for every 5 epoch
    if epoch % 5 == 0:
        meanF1, maxF1, mae1 = eval_model(net, tes_dataloader_davis, test_num_davis)
        meanF2, maxF2, mae2 = eval_model(net, tes_dataloader_davsod, test_num_davsod)
        print('epoch:', epoch)
        print('[meanF] davis: %.4f davsod: %.4f' % (meanF1,meanF2))
        print('[maxF] davis: %.4f davsod: %.4f' % (maxF1,maxF2))
        print('[mae] davis: %.4f davsod: %.4f' % (mae1,mae2))

        writer.add_scalar('Test/meanF_davis', meanF1, epoch)
        writer.add_scalar('Test/maxF_davis', maxF1, epoch)
        writer.add_scalar('Test/mae_davis', mae1, epoch)
        writer.add_scalar('Test/meanF_davsod', meanF2, epoch)
        writer.add_scalar('Test/maxF_davsod', maxF2, epoch)
        writer.add_scalar('Test/mae_davsod', mae2, epoch)
        writer.file_writer.flush()

        # save the best
        if maxF1 > best_mf or maxF1 > 0.90:
            print('best maxF: %.4f -> %.4f' % (best_mf, maxF1))
            save_dir = model_dir + "/epoch_%d_davis_%.4f_davsod_%.4f.pth" % (epoch, maxF1, maxF2)
            if args.device_num != 1:
                torch.save(net.module.state_dict(), save_dir)
            else:
                torch.save(net.state_dict(), save_dir)
            print('model save at', save_dir)
            best_mf = maxF1
