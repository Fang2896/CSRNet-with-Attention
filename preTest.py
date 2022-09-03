from model import CSRNet, CSRNet_senet, CSRNet_cbam, CSRNet_CA
import dataset
import json
from torchvision import  transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from model import SpatialAttention

# 读入json list。所有文件的路径
with open('part_A_train.json', 'r') as outfile:
    train_list = json.load(outfile)

# 加载模型
model = CSRNet_cbam()
model = model

# tensorboard
writer = SummaryWriter('logs/preTest/')

# 加载数据
# dummy_input = torch.rand(1, 3, 1024, 764)

# train_loader = DataLoader(
#         dataset.listDataset(train_list,
#                        shuffle=True,
#                        transform=transforms.Compose([
#                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
#                    ]),
#                        train=True,
#                        seen=model.seen,
#                        batch_size=1,
#                        num_workers=4),
#         batch_size=1)
for name, m in model.named_modules():
    if isinstance(m, SpatialAttention):
        print(m)
        print("================")