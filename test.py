import os
import glob
from sklearn.metrics import mean_squared_error,mean_absolute_error
import torchvision.transforms.functional as F
from image import *
from model import CSRNet
from model import CSRNet_senet
from model import CSRNet_CA
from model import CSRNet_cbam
import torch


from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])

root = '../../Dataset'

# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
for p in img_paths:
    print(p)

# test
model = CSRNet_CA()
model = model.cuda()


# 保存好的模型
checkpoint = torch.load('trainedModel/shtech_partB_CA/CA_model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

pred = []
gt = []
for i in range(len(img_paths)):
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    img[0, :, :] = img[0, :, :] - 92.8207477031
    img[1, :, :] = img[1, :, :] - 95.2757037428
    img[2, :, :] = img[2, :, :] - 104.877445883
    img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda() # 原本被注释了
    gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    pred.append(output.data.cpu().numpy().sum())
    gt.append(np.sum(groundtruth))
    # mae += abs(output.detach().cpu().sum().numpy() - np.sum(groundtruth))
    # print(i, mae)
# print("Average MAE: ", mae / len(img_paths))
mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print('MAE: ',mae)
print('RMSE: ',rmse)