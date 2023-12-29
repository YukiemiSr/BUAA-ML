
NUM_CLASSES = 3
LABELS = [
    "setosa","versicolour","virginica"
]
LABEL_MAP = {
    0: "setosa", 1: "versicolour", 2: "virginica"
}

from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
#    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

from torchvision.datasets import ImageFolder, VisionDataset
from torch.utils.data import DataLoader
import os

data_path = './Iris_data' #@param
batch_size = 16 #@param
num_workers = 0 #@param
train_path = (os.path.join(data_path, 'train'))
#用Transforms将图片进行裁剪，并转化为Tensor；用ImageFolder即可自然完成样本和标签的对应
train_dataset = ImageFolder(
    train_path,
    transform_train)
#用DataLoader将数据分批次加载
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,shuffle=True)

import os
import PIL
import torchvision.transforms as transforms
from PIL import Image

def convert_to_rgb(img_path):
    """
    将图片路径转换成 RGB 格式。

    Args:
        img_path: 图片路径。

    Returns:
        转换后的图片路径。
    """

    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    img_path_rgb = img_path.replace(".jpg", ".rgb")
    img.save(img_path_rgb)
    return img_path_rgb

def load_test_set(test_dir):
    # 获取测试集文件夹中的所有图片路径
    img_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]

    # 定义数据转换器
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # 加载测试集
    test_set = []
    for img_path in img_paths:
        img_path = convert_to_rgb(img_path)
        # 读取图片
        img = Image.open(img_path)

        # 数据转换
        img = transform_test(img)[:3]

        # 把len变为2
#        img = img.unsqueeze(0)

        # 添加到测试集中
        test_set.append(img)

    return test_set

test_path = (os.path.join(data_path, 'test'))

test_dataset = load_test_set(test_path)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    )




# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import torch
import torchvision
from torchvision import transforms

# 导入模型
model = torchvision.models.resnet50(pretrained = True).to(device)

# 获取模型的输出特征维度
num_features = model.fc.in_features
# 修改模型的输出层
num_classes = 3
model.fc = nn.Linear(num_features, num_classes)
model.fc.to('cuda:0')

model.train()
learning_rate = 1e-5 #@param
batch_size = 64 #@param
epochs = 30 #@param
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f'):
      self.name = name
      self.fmt = fmt
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

  def __str__(self):
      fmtstr = '{name} {avg' + self.fmt + '}'
      return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')

import time
import json

best_accuracy = 0
start = time.time()
for i in range(101):
  model.train()
  for batch, (X, y) in enumerate(train_loader):
    X = X.to(device)
    y = y.to(device)
    data_time.update(time.time() - start)
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), X.size(0))

  batch_time.update(time.time() - start)
  start = time.time()

  print(f"Epoch:{i + 1}: {batch_time}, {losses}")
  model.eval()
  # 创建一个字典来存储测试结果
  test_results = {}
  def get_key(dic,value):
    return [k for k,v in dic.items() if v == value]
  transform_test = transforms.Compose([
          transforms.Resize((224,224)),
          transforms.ToTensor(),
          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
  ])
print("训练完成")
LABEL_MAP = {
    0: "setosa", 1: "versicolour", 2: "virginica"
}
test_path = './Iris_data/test/'
import pandas as pd
test = pd.read_csv('test.csv',header=None)
with torch.no_grad():
    with open('answer.txt', 'w') as file:
        for j in range(0, 156):
            name = test.iloc[j, 0]
            img_path = os.path.join(test_path, name)
            image = Image.open(img_path)
            image = transform_test(image)[:3].unsqueeze(0)
            image = image.to(device)
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.max(probabilities, 1)[1]
            label = LABEL_MAP[predicted.tolist()[0]]
            print(label)
            file.write(label + '\n')