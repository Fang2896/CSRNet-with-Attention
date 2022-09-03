from model import CSRNet, CSRNet_senet, CSRNet_cbam
import dataset
import json
from torchvision import  transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

# 读入json list。所有文件的路径
with open('part_A_train_with_val.json', 'r') as outfile:
    train_list = json.load(outfile)

# 加载数据
train_loader = DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train=True,
                       seen=0,
                       batch_size=1,
                       num_workers=4),
        batch_size=1)

model = CSRNet_cbam()


with SummaryWriter(comment='CSRNet_with_CBAM_attention') as w:
    w.add_graph(model, [dummy_input])

    


w.close()
