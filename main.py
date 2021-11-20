from torch.utils import data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import dct2, idct2, init_mask
from torchvision.transforms import transforms
from argparse import PARSER


class FSDRDataset(Dataset):
    '''
        mask: mask applied to the dct of each image
    '''
    def __init__(self, mask, data_df, transform=None):
        '''
        mask: mask whose shape is same as shape of images in the dataset
        data_df: csv containing file name and corresponding label
        transform = transformation to be applied to images 
        '''
        self.mask = mask
        self.transform = transform
        self.data_df = data_df

    def get_masked_image(self, dct, mask):
        masked_dct = dct*mask
        masked_image = idct2(masked_dct)
        return masked_image

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        image = Image.open(self.data_df[index].convert('RGB'))
        image = self.get_masked_image(image, self.mask)
        label = self.data_df['label'][index]
        return image, label 

transpose = transforms.Compose(transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
shape = [3,32,32]
mask = init_mask(shape=shape)

dataset = FSDRDataset(mask, data_df=)
