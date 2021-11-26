from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
from torchvision.datasets import ImageFolder
from argparse import PARSER
from utils import init_mask
from torch.utils.data import DataLoader
import argparse
from models import ClassificationModel, SegmentationModel
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser(description='Auto FSDR Training')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
# parser.add_argument()

args = parser.parse_args()

writer = SummaryWriter()

transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
shape = [3,32,32]
mask = init_mask(shape=shape)
train_root = r'C:\Users\sanje\Documents\Projects\mlls_project\datasets\CIFAR10\test' 
# val_root = ''

train_dataset = ImageFolder(train_root, transform=transform)
# val_dataset = ImageFolder(val_root)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# input_shape = [args.batch_size] + shape
input_shape = shape
# print(input_shape)
model = ClassificationModel(input_shape=input_shape, dim=10)

# Training Specifics
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()

for epoch in range(args.num_epochs):
    epoch_loss = []
    for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    avg_loss = np.mean(epoch_loss)
    print('[%d] loss: %0.3f' % (epoch, avg_loss))
    writer.add_scalar('loss', avg_loss)
