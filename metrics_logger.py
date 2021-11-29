import pickle
import matplotlib.pyplot as plt
import os
from time import time

from torch.serialization import save

class MetricsLogger:
    def __init__(self, model_details):
        self.metrics_dict = {}
        self.model_details=model_details
        self.metrics_dict['Model Details'] = self.model_details

        pass
    
    def add_metric(self, name: str, x_val, y_val):
        if not name in self.metrics_dict.keys():
            self.metrics_dict[name] = {}
        self.metrics_dict[name][x_val] = y_val
        
    def plot_metrics(self, which='all', save_dir='./metrics/'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if which=='all':
            metric_names = self.metrics_dict.keys()
        
        for metric_name in metric_names:
            fname = os.path.join(save_dir, metric_name)
            d = self.metrics_dict[metric_name]
            x = list(d.keys())
            y = list(d.values())
            plt.plot(x, y)
            plt.savefig(fname=fname, facecolor='white', )
    
    def save_dict(self, save_path='auto'):
        if save_path=='auto':
            save_path = './metrics/metrics.pickle'+ str(time())
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump(self.metrics_dict, f)


