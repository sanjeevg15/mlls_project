from torch.cuda import current_blas_handle
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils import data
# import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
from torchvision.datasets import ImageFolder
from argparse import PARSER
from utils import init_mask
from torch.utils.data import DataLoader
import argparse
import os
from models import *
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Auto FSDR Training')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--data_root', default=r'C:\Users\sanje\Documents\Projects\mlls_project\datasets\CIFAR10', type=str)
parser.add_argument('--target_domain', default=-1, type=int, help='Index of the target domain. Remaining domains will be treated as source domains')
parser.add_argument('--input_shape', default=224, type=int, help='Resize all images to this shape')
parser.add_argument('--save_dir', default='../ckpt', type=str, help='Directory to save model checkpoints')
parser.add_argument('--log_dir', default='../logs', type=str, help='Directory to save tensorboard logs')
parser.add_argument('--save_ckpt', default='best', type=str, help='Checkpoint saving strategy')
args = parser.parse_args()

transform = T.Compose([T.Resize((224,224)), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

domains = [i for i in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, i))]
classes = os.listdir(os.path.join(args.data_root, domains[0]))
n_classes = len(classes)
images = os.listdir(os.path.join(args.data_root, domains[0], classes[0]))
input_shape = (args.input_shape, args.input_shape)

assert(args.save_ckpt in ['best', 'last', 'all']) # checkpoint saving strategy

domain_datasets = {}
for domain in domains:
    domain_datasets[domain] = ImageFolder(os.path.join(args.data_root, domain), transform=transform)

# domain to be considered as target (testing) for domain generalisation setting; rest are source (training) domains.

concat_datasets = []
for domain in domain_datasets:
    if domain!=domains[args.target_domain]:
        concat_datasets.append(domain_datasets[domain])
source_dataset = data.ConcatDataset(concat_datasets)
target_dataset = domain_datasets[domains[args.target_domain]]

source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=args.batch_size)

model = ClassificationModel(input_shape=input_shape, dim=n_classes, use_resnet=True, resnet_type='resnet18').to(device) # resnet_type can be: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'


# Training Specifics
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()
if args.save_ckpt=='best':
    best_test_accuracy = 0.0

# Logging Specifics 
writer = SummaryWriter(args.log_dir)

total_iters = 0
mask_weights1 = model.mask.weights.clone().cpu().data.numpy()
mask_weigths_diff = []
for epoch in range(args.num_epochs):
    epoch_loss = []
    model.train()

    for i, (inputs, labels) in tqdm(enumerate(source_loader)):
        optimizer.zero_grad()
        outputs = model(inputs.to(device)) # [64 * 7]
        loss = loss_fn(outputs, labels.to(device))

        total_iters+=1
        writer.add_scalar('loss/c_entropy_per_iteration', loss.item(), total_iters)
        epoch_loss.append(loss.item())
        print("{}-{}: Iteration Loss: {}".format(epoch+1, i+1, loss.item()))
        loss.backward()
        optimizer.step()
        mask_weights2 = model.mask.weights.clone().cpu().data.numpy()
        mask_weigths_diff.append(np.linalg.norm(mask_weights2-mask_weights1))
        current_weigths_diff = np.linalg.norm(mask_weights2-mask_weights1)
        writer.add_scalar('freq_mask_weigths_norm_diff', current_weigths_diff, total_iters)
        mask_weights1 = mask_weights2

    # logging mean epoch loss
    avg_loss = np.mean(epoch_loss)
    writer.add_scalar('loss/mean_c_entropy_per_epoch', avg_loss, epoch+1)
    print('[%d] mean epoch loss: %0.3f' % (epoch+1, avg_loss))

    model.eval()

    # logging accuracy on complete train set
    correct = 0
    for i, (inputs, labels) in enumerate(source_loader):
        outputs = model(inputs.to(device))
        _, outputs = torch.max(outputs, dim=1)
        correct += (outputs == labels.to(device)).float().sum()
    train_accuracy = 100 * correct / len(source_dataset)
    print("Accuracy on train-dataset:", train_accuracy.item())
    writer.add_scalar('accuracy/train_set_accuracy_per_epoch', train_accuracy.item(), epoch+1)

    # logging accuracy on complete test set
    correct = 0
    for i, (inputs, labels) in enumerate(target_loader):
        outputs = model(inputs.to(device))
        _, outputs = torch.max(outputs, dim=1)
        correct += (outputs == labels.to(device)).float().sum()
    test_accuracy = 100 * correct / len(target_dataset)
    print("Accuracy on test-dataset[{}]:".format(domains[args.target_domain]), test_accuracy.item())
    writer.add_scalar('accuracy/test_set_accuracy_per_epoch:{}'.format(domains[args.target_domain]), test_accuracy.item(), epoch+1)
    
    # logging histogram of mask values
    writer.add_histogram('histogram/mask', torch.sigmoid(model.mask.weights).
    clone().cpu().data.numpy(), epoch+1)

    # saving the model
    if args.save_ckpt=='last':
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'ckpt_last.pt'))
    elif args.save_ckpt=='best':
        if test_accuracy > best_test_accuracy:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'ckpt_best.pt'))
            best_test_accuracy = test_accuracy
    else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'ckpt_{}.pt'.format(epoch+1)))
    


