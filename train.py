from main import FSDRDataset
import numpy as np
import argparse
import pickle
import torch
import os
import random
import torch.optim as optim
import torch.nn as nn

from models import resnet18, Model1
from torch.utils.data import DataLoader
from utils import init_mask
from torchvision import transforms, datasets


lr_mask = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='', type=str,
                    help='image dataset folder')
parser.add_argument('--num_epochs', default=20, type=int,
                    help='number of epochs to train the model for')
parser.add_argument('--batch_size', default=64,
                    type=int, help='training batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training')
# parser.add_argument('--seed', default=42, type=int, help='random seed')
args = parser.parse_args()

if not os.path.exists('trained_models/'):
    os.makedirs('trained_models/')

MODEL_NAME = 'model1'
MODEL_PATH = 'trained_models/' + MODEL_NAME + '.pt'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    normalizer
])

model = Model1()
train_dataset = ImageDataset(transform=data_transform)
# train_dataset = FSDRDataset(transform=data_transform)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
cross_entropy = nn.CrossEntropyLoss()
params = model.parameters()

for i, param in enumerate(params):
    if i == 0:
        param.requires_grad = True
    else:
        param.requires_grad = True

param.grad = get_mask_grad()

optimizer = optim.Adam(params, lr=args.lr)
model.train()

for epoch in range(args.num_epochs):
    epoch_loss = []
    for inputs, labels in train_loader:
        params_optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = cross_entropy(outputs, labels)
        epoch_loss.append(loss.item())
        loss.backward()
        params_optimizer.step()
    avg_loss = np.mean(epoch_loss)
    print('[%d] loss: %.4f' % (epoch, avg_loss))

    torch.save(model.state_dict(), MODEL_PATH)
