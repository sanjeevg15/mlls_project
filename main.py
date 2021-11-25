from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils import data
# import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
from torchvision.datasets import ImageFolder
from argparse import PARSER
from utils import init_mask
from torch.utils.data import DataLoader
import argparse
from models import *
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Auto FSDR Training')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
# parser.add_argument()

args = parser.parse_args()

transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
shape = [3,227,227]
mask = init_mask(shape=shape)
train_root = r'/raid/home/jsingh319/codes/dataset/pacs/pacs_data_original/art_painting' 
# val_root = ''

train_dataset = ImageFolder(train_root, transform=transform)
# val_dataset = ImageFolder(val_root)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# input_shape = [args.batch_size] + shape
input_shape = shape
# print(input_shape)
model = Model1(input_shape=input_shape, dim=7, use_resnet=True).cuda()

# Training Specifics
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()

# Logging Specifics 
writer = SummaryWriter('../logs')

total_iters = 0
for epoch in range(args.num_epochs):
    epoch_loss = []
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs.cuda()) # [64 * 7]
        loss = loss_fn(outputs, labels.cuda())

        total_iters+=1
        writer.add_scalar('loss/c_entropy_per_iteration', loss.item(), total_iters)
        epoch_loss.append(loss.item())
        print("{}-{}: Iteration Loss: {}".format(epoch+1, i+1, loss.item()))

        loss.backward()
        optimizer.step()

    # logging mean epoch loss
    avg_loss = np.mean(epoch_loss)
    writer.add_scalar('loss/mean_c_entropy_per_epoch', avg_loss, epoch+1)
    print('[%d] mean epoch loss: %0.3f' % (epoch+1, avg_loss))

    # logging accuracy on complete train set
    model.eval()
    correct = 0
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs.cuda())
        _, outputs = torch.max(outputs, dim=1)
        correct += (outputs == labels.cuda()).float().sum()
    train_accuracy = 100 * correct / len(train_dataset)
    print("Accuracy on train-dataset:", train_accuracy.item())
    writer.add_scalar('accuracy/train_set_accuracy_per_epoch', train_accuracy.item(), epoch+1)

    # logging histogram of mask values
    writer.add_histogram('histogram/mask', torch.sigmoid(model.mask.weights).clone().cpu().data.numpy(), epoch+1)
    


