# Get the average DCT over all images in a particular domain. Optionally, for a particular class of the domain, or particular class across domains
from utils import dct_2d
from PIL import Image
import numpy as np
import os
import pickle

class DCTAverager():
    def __init__(self, root, save_dir = './avg_dir_results'):
        self.root = root
        self.save_dir = save_dir

    def get_avg_dct(self, domains = 'all', target_shape=(224,224)):
        if domains == 'all':
            domains = os.listdir(self.root)
        for domain in domains:
            self.avg_dct = np.zeros(target_shape)
            image_folder = os.path.join(domain)
            for image_name in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image_name)
                img = np.array(Image.read(image_path).resize(target_shape))
                self.avg_dct += dct_2d(img)
            self.avg_dct/= len(os.listdir(image_folder))
            
            


        