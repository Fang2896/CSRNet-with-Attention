import os

# 导入各个网络
from model import CSRNet
from model import CSRNet_senet
from model import CSRNet_cbam
from model import CSRNet_CA

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import  transforms
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

import argparse
import json
import dataset
import time

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')
parser.add_argument('att', metavar='Attention', type=str,
                    help='add attention mechanism: none & senet & cbam & ca')
parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')
parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6

    # 可视化
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 300
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(int(args.seed))

    # 这个dummy_input用来生成
    dummy_input = torch.rand(1, 3, 1024, 764).cuda()
    # 根据参数来选择网络，是有注意力还是没有注意力
    if args.att == 'senet':
        netflag = 'SENET'
        model = CSRNet_senet()
        model = model.cuda()
        writer = SummaryWriter('logs/shtech_B/SEnet/')
        writer.add_graph(model, [dummy_input]) # 添加模型结构
    elif args.att == 'cbam':
        netflag = 'CBAM'
        model = CSRNet_cbam()
        model = model.cuda()
        writer = SummaryWriter('logs/shtech_B/CBAM/')
        writer.add_graph(model, input_to_model=dummy_input)
    elif args.att == 'ca':
        netflag = 'CA'
        model = CSRNet_CA()
        model = model.cuda()
        writer = SummaryWriter('logs/shtech_B/CA/')
        writer.add_graph(model, input_to_model=dummy_input)
    else:
        netflag = 'CSRNet'
        model = CSRNet()
        model = model.cuda()
        writer = SummaryWriter('logs/shtech_B/CSRnet/')
        writer.add_graph(model, input_to_model=dummy_input)

    criterion = nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    # 如果有之前训练的，继续训练
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        epoch_time_start = time.time()

        adjust_learning_rate(optimizer, epoch)
        # 训练
        train(train_list, model, criterion, optimizer, epoch)
        # 验证
        prec1 = validate(val_list, model, criterion)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)

        # 写入tensorboard
        writer.add_scalar(tag='train/prec '+netflag,
                          scalar_value=prec1,
                          global_step=epoch)
        writer.add_scalar(tag='train/bestMAE '+netflag,
                          scalar_value=best_prec1,
                          global_step=epoch)
        writer.add_scalar(tag='train/learningRate '+ netflag,
                          scalar_value=args.lr,
                          global_step=epoch)    # 学习率可视化
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

        epoch_time_end = time.time()
        print('* Epoch Time {epoch_time:.2f}'
              .format(epoch_time=epoch_time_end - epoch_time_start))

    writer.close()


def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 训练！
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        
        
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = DataLoader(
        dataset.listDataset(val_list,
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                       ]),  train=False),
        batch_size=args.batch_size)
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))
    return mae    


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


if __name__ == '__main__':
    main()        