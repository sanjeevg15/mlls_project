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
from metrics_logger import MetricsLogger

def train_model(model, num_epochs, optimizers, loss_fn, train_regime='normal', log_dir='./logs', save_dir='./ckpt', model_details_dict={}, save_ckpt='best', test=False):
    # Initialize variables to log metrics
    total_iters = 0
    mask_weights1 = model.mask.weights.clone().cpu().data.numpy()
    mask_weigths_diff = []
    optimizer1, optimizer2 = optimizers

    # Create logger to log metrics
    logger = MetricsLogger(model_details = model_details_dict)
    writer = SummaryWriter(log_dir)

    if save_ckpt=='best':
        best_test_accuracy = 0.0 

    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        if train_regime == 'alternating':
            if model.mask.weights.requires_grad:
                for parameter in model.parameters():
                    parameter.requires_grad = True
                    model.mask.weights.requires_grad = False
            else:
                for parameter in model.parameters():
                    parameter.requires_grad = False
                    model.mask.weights.requires_grad = True
            

        for i, (inputs, labels) in tqdm(enumerate(source_loader)):
            optimizer1.zero_grad()
            outputs = model(inputs.to(device)) # [64 * 7]
            loss = loss_fn(outputs, labels.to(device))

            total_iters+=1
            writer.add_scalar('loss/c_entropy_per_iteration', loss.item(), total_iters)
            logger.add_metric('loss', total_iters,loss.item())
            loss.add_metric('freq_mask', total_iters, model.mask.weights)
            epoch_loss.append(loss.item())
            print("{}-{}: Iteration Loss: {}".format(epoch+1, i+1, loss.item()))
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            mask_weights2 = model.mask.weights.clone().cpu().data.numpy()
            mask_weigths_diff.append(np.linalg.norm(mask_weights2-mask_weights1))
            current_weigths_diff = np.linalg.norm(mask_weights2-mask_weights1)
            writer.add_scalar('freq_mask_weigths_norm_diff', current_weigths_diff, total_iters)
            logger.add_metric('freq_mask_change', total_iters, current_weigths_diff)
            logger.add_metric('mask_weights_grad', total_iters, model.mask.weights.grad)
            mask_weights1 = mask_weights2
            if test:
                if i == 5:
                    break

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
            if test:
                if i == 5:
                    break
        train_accuracy = 100 * correct / len(source_dataset)
        print("Accuracy on train-dataset:", train_accuracy.item())
        logger.add_metric('train_accuracy', epoch, train_accuracy.item())

        writer.add_scalar('accuracy/train_set_accuracy_per_epoch', train_accuracy.item(), epoch+1)

        # logging accuracy on complete test set
        correct = 0
        for i, (inputs, labels) in enumerate(target_loader):
            outputs = model(inputs.to(device))
            _, outputs = torch.max(outputs, dim=1)
            correct += (outputs == labels.to(device)).float().sum()
            if test:
                if i == 5:
                    break
        test_accuracy = 100 * correct / len(target_dataset)
        print("Accuracy on test-dataset[{}]:".format(domains[target_domain]), test_accuracy.item())
        logger.add_metric('test_accuracy', epoch, test_accuracy.item())

        writer.add_scalar('accuracy/test_set_accuracy_per_epoch:{}'.format(domains[target_domain]), test_accuracy.item(), epoch+1)

        # logging histogram of mask values
        writer.add_histogram('histogram/mask', torch.sigmoid(model.mask.weights).
        clone().cpu().data.numpy(), epoch+1)

        # saving the model
        if save_ckpt=='last':
            torch.save(model.state_dict(), os.path.join(save_dir, 'ckpt_last.pt'))
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
        elif save_ckpt=='best':
            if test_accuracy > best_test_accuracy:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ckpt_best.pt'))
                best_test_accuracy = test_accuracy
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ckpt_{}.pt'.format(epoch+1)))
    
    # domain_best_accuracies[domains[target_domain]] = best_test_accuracy.item()
    logger.add_metric('best_target_accuracies', domains[target_domain], best_test_accuracy)
    logger.save_dict()
    
    # Logging Specifics 

if __name__ == '__main__':

    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Auto FSDR Training')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--target_domain', default=0, type=int, help='Index of the target domain. Remaining domains will be treated as source domains')
    parser.add_argument('--all_target_domain', default=False, action='store_true', help='Using on each domain as testing domain one at a time')
    parser.add_argument('--input_shape', default=224, type=int, help='Resize all images to this shape')
    parser.add_argument('--no_fq_mask', default=False, action='store_true', help='Turn off the frequency mask')
    parser.add_argument('--lr1', default=1e-4, type=float, help='Learning rate for other layers training')
    parser.add_argument('--lr2', default=1e-4, type=float, help='Learning rate for freq mask training')
    parser.add_argument('--save_dir', default='./ckpt', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', default='./logs', type=str, help='Directory to save tensorboard logs')
    parser.add_argument('--save_ckpt', default='best', type=str, help='Checkpoint saving strategy')
    parser.add_argument('--train_regime', default='normal', type=str, help = " 'normal' or 'alternating'. If normal, all layers are trained simultaneously. If alternating, frequency mask & remaining layers are trained alternatively keeping one part frozen every time")
    parser.add_argument('--initialization', default='ones', type=str, help="'ones' or 'random_normal' or 'xavier' initialization for the frequency mask")
    parser.add_argument('--test', default=False, action='store_true', help='Quick test for debuggin purposes')
    args = parser.parse_args()


    # Training Specifics
    num_epochs = args.num_epochs
    train_regime = args.train_regime
    initialization = args.initialization

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #Select device

    transform = T.Compose([T.Resize((224,224)), 
                T.ToTensor(), 
                T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]) #Transform every image as per the mentioned transform

    domains = [i for i in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, i))] # Identify domains from dataset
    classes = os.listdir(os.path.join(args.data_root, domains[0])) #Identify classses
    n_classes = len(classes) #Number of classes to decide final layer dimensions

    assert(args.save_ckpt in ['best', 'last', 'all']) # checkpoint saving strategy

    # domain to be considered as target (testing) for domain generalisation setting; rest are source (training) domains.
    domain_datasets = {}
    for domain in domains:
        domain_datasets[domain] = ImageFolder(os.path.join(args.data_root, domain), transform=transform)

    assert(args.target_domain>=0)
    if args.all_target_domain:
        min_target_domain, max_target_domain = 0, len(domain_datasets)
    else:
        min_target_domain, max_target_domain = args.target_domain, args.target_domain+1

    # domain_best_accuracies = {}
    for target_domain in range(min_target_domain, max_target_domain):

        concat_datasets = []
        for domain in domain_datasets:
            if domain!=domains[target_domain]:
                concat_datasets.append(domain_datasets[domain])
        source_dataset = data.ConcatDataset(concat_datasets)
        target_dataset = domain_datasets[domains[target_domain]]

        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size)

        model = ClassificationModel(input_shape=args.input_shape, dim=n_classes, use_resnet=True, resnet_type='resnet18', no_fq_mask=args.no_fq_mask, mask_initialization=initialization).to(device) # resnet_type can be: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'

        # Training Specifics
        params1 = (i for i in list(model.parameters())[1:])
        params2 = (i for i in list(model.parameters())[0])
        optimizer1 = optim.Adam(params=params1, lr=args.lr1)
        optimizer2 = optim.Adam(params=params2, lr=args.lr2)
        optimizers = [optimizer1, optimizer2]
        loss_fn = CrossEntropyLoss()


        # create checkpoint and log dir
        log_dir = os.path.join(args.log_dir, domains[target_domain])
        save_dir = os.path.join(args.save_dir, domains[target_domain])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        # Training Details
        model_details_dict = {'Model Name': model.name, 'Target Domain':domains[target_domain], 'Freq Mask': not(args.no_fq_mask), 'Optimizers': optimizers,'Num Epochs': num_epochs, 'loss_fn': loss_fn, 'Initialization':initialization}
        train_model(model, num_epochs, optimizers, loss_fn, train_regime, log_dir, save_dir, model_details_dict, test=args.test) #Initiate training 




        # Print/Save metrics
        # print("Best test accuracies on all target domains:", domain_best_accuracies)
        # print("Average test accuracies across domains:", sum(domain_best_accuracies.values()) / len(domain_best_accuracies))
        


