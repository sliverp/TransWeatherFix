import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
# from utils import to_psnr, print_log, validation, adjust_learning_rate
from utils import to_psnr, print_log, validation
import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

from transweather_model import Transweather

import time

parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[480, 720], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment',default="runs" ,type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)
args = parser.parse_args()
labeled_name = 'allweather.txt'
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
train_dataset_rate = 0.7
train_data_dir = '/data/allweather/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
full_dataset = TrainData(crop_size, train_data_dir,labeled_name)
train_size = int(train_dataset_rate * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
lbl_train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(test_dataset, batch_size=train_batch_size,  num_workers=8)

seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-img_path', help='', default='./0.png', type=str)
parser.add_argument('-pt', help='',default='./runs1/best_all' ,type=str)
args = parser.parse_args()


# net = Transweather()
# net.load_state_dict(torch.load(args.pt),strict=False)
net = torch.load(args.pt)
net = net.to(device)
net.eval()

def save_image(images):
    trans = T.Compose([
        T.ToPILImage(),
        # T.Resize((360,640)),
    ])
    
    pic = trans(images[0])
    pic.save('result.jpg')


def main():
    # print(validation(net, val_data_loader, device, exp_name))
    trans = T.Compose([
        # T.Resize((256,256)),
        T.ToTensor()
    ])
    img = trans(Image.open(args.img_path))
    img = img.unsqueeze(0)
    img = img.to(device)
    
    start = time.clock()
    handled = net(img)
    end = time.clock()
    print(end-start)

    save_image(handled)

if __name__ == '__main__':
    main()