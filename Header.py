# Python Header - Import dir

import os
import argparse
import numpy as np
import random
import shutil
import pickle
import matplotlib.pyplot as plt
import warnings
import math

# for no warning sign
warnings.filterwarnings('ignore')

# Check os type
if(os.name == 'nt' or os.name == 'posix'): # windows or ubuntu linux
    import tensorflow as tf
    from tensorflow.contrib.keras import backend as K
    from tensorflow.contrib.keras import layers as l
    from tensorflow.contrib.keras import models as m
    from tensorflow.contrib.keras import optimizers as op
    from tensorflow.contrib.keras import callbacks as call
    from tensorflow.contrib.keras import regularizers as rz
    from tensorflow.contrib.keras import initializers as lz
    from tensorflow.contrib.keras import utils as ut
    from tensorflow.contrib.keras import applications as ap
    from tensorflow.contrib.keras import activations as act
    # Randomness setting
    seed = int(os.getenv("SEED", 12))
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
else:
    print('The os type is not windows or linux. This code supports only windows and linux')
    exit(1)


# Temperature storage
temp_dir = "./temp/"
temp_file = "./temp/temperature.pickle"

# Global Function

# Dir maker
def check_and_makedir(path,choice=False):
    if(choice==True):
        shutil.rmtree(path, ignore_errors=True)
    if not os.path.isdir(path):
        os.makedirs(path)

# For boolean
def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False

# GPU SETUP
def set_up(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

def Set_Model_Info():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # for hardware acceleration
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--model_usage', type=str, default="teacher", choices=["teacher", "student"])

    # Language
    parser.add_argument('--model_lang',type=str,default="keras",choices=["keras","tensorflow"])

    return parser


def Set_Parameter(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Temperature
    parser.add_argument('--temperature', type=float, default=1.0)

    # Set epoch and stop point criterion
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--max_overfit',type=int,default=50)

    # Model Type
    parser.add_argument('--model_type',type=str,default="classifier",choices=["classifier", "regression"])

    ## Learning Parameters ##

    # Set learning rate
    parser.add_argument('--min_learning_rate',type=float,default=0.001)
    parser.add_argument('--learning_rate_increment',type=float,default=10)
    parser.add_argument('--max_learning_rate',type=float,default=0.001)

    # Set batch size
    parser.add_argument('--min_batch', type=int, default=128)
    parser.add_argument('--batch_increment', type=int, default=2)
    parser.add_argument('--max_batch', type=int, default=128)

    if(args.model_usage == "teacher"):
        # Learning parameter
        # parser.add_argument('--dropout',type=float,default=0.3)

        # Save path
        parser.add_argument('--dir',type=str,default="./Teacher_model/")
        parser.add_argument('--checkpoint',type=str,default="./Teacher_model/checkpoint/") # checkpoint weight saved
        parser.add_argument('--model',type=str,default="./Teacher_model/model/") # model saved
        parser.add_argument('--final',type=str,default="./Teacher_model/final/") # final weight saved
        parser.add_argument('--log',type=str,default="./Teacher_model/log/") # log dir for softy data
        
    else:

        ## Model Parameters ##
    
        # DNN Node Setting
        parser.add_argument('--min_node',type=int,default=16)
        parser.add_argument('--node_increment',type=int,default=2)
        parser.add_argument('--max_node',type=int,default=256)

        # Layer Setting
        parser.add_argument('--min_layer',type=int,default = 1)
        parser.add_argument('--layer_increment',type=int,default = 1)
        parser.add_argument('--max_layer',type=int,default = 4)

        # Save path
        parser.add_argument('--dir',type=str,default="./Student_model/")
        parser.add_argument('--checkpoint',type=str,default="./Student_model/checkpoint/") # checkpoint weight saved
        parser.add_argument('--model',type=str,default="./Student_model/model/") # model saved
        parser.add_argument('--final',type=str,default="./Student_model/final/") # final weight saved
        parser.add_argument('--log',type=str,default="./Student_model/log/") # log dir for recording models
        parser.add_argument('--tmp',type=str,default="./Student_model/tmp/") # tmp dir for simulating single model
        parser.add_argument('--softy_file',type=str,default="./Teacher_model/log/soft_y.pickle") # log dir for softy data
        parser.add_argument('--teacher_acc_file',type=str,default="./Teacher_model/log/Accuracy.pickle") # log dir for teacher accuracy

    return parser


