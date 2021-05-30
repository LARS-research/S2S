import logging
import os
import datetime
import random

from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable



class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def logger_init(args):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if args.log_to_file:
        log_filename = os.path.join(args.log_dir, args.log_prefix+datetime.datetime.now().strftime("%m%d%H%M%S"))
        logging.getLogget().addHandler(logging.FileHandler(log_filename))


def plot_config(args):
    out_str = "\noptim:{}, lr:{}, in_dropout:{}, hd_dropout:{}, d:{}, decay_rate:{}, train_batch_size:{}, valid_batch_size:{}, M_val:{}, n_epoch:{}\n".format(
            args.optim, args.lr, args.input_dropout, args.hidden_dropout, args.n_dim, args.decay_rate, args.n_batch, args.valid_batch, args.M_val, args.n_epoch)
    print(out_str)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]
        
def gen_struct(num):
    struct = []
    for i in range(num):
        if i < 4:
            struct.append(random.randint(0,3))      #t
        else:
            struct.append(random.randint(0,3))      #h
            struct.append(random.randint(0,3))      #t
            struct.append(random.randint(-1,1))  #1
    return struct

def default_search_hyper(args):
    
    if args.dataset == 'WN18RR':
        args.lr = 0.47102439590590006
        #args.lamb = 5.919173541218532e-05      # searched
        #args.lamb = 1.8402163609403787e-05      # SimplE
        args.lamb = 0.0002204803280058515       # ComplEx
        args.decay_rate = 0.9903840888956048
        
        args.n_dim = 512
        args.n_batch = 512
        args.n_epoch = 3
        args.epoch_per_test = 3 #20
        
    elif args.dataset == 'FB15K237':
        args.lr = 0.1783468990895745
        args.lamb = 0.0025173667237246883
        args.decay_rate = 0.9915158217372417
        
        args.n_batch = 512
        args.n_dim = 2048
        args.n_epoch = 500 
        args.epoch_per_test = 5 #25
        
        #args.n_dim = 512
        
    elif args.dataset == 'WN18':
        args.lr = 0.10926076305780041
        args.lamb = 0.0003244851835920663
        args.decay_rate = 0.9908870395744
        args.n_batch = 256
        args.n_dim = 1024 #AutoSF searched
        args.n_epoch = 400 #AutoSF searched
        args.epoch_per_test = 4 #20
        
        #args.n_dim = 512
        
    elif args.dataset == 'FB15K':
        args.lr = 0.7040329784234945
        args.lamb = 3.49037818818688153e-5
        args.decay_rate = 0.9909065915902778
        args.n_batch = 512
        args.n_epoch = 700 #AutoSF searched
        args.n_dim = 2048 #AutoSF searched
        args.epoch_per_test = 7 #50
        
        #args.n_dim = 512
        
    elif args.dataset == 'YAGO':
        args.lr = 0.9513908770180219
        args.lamb = 0.00021779088577909324
        args.decay_rate = 0.9914972709145934
        args.n_batch = 512 # 2048
        #args.n_dim = 1024
        #args.n_epoch = 400
        args.n_dim = 512
        args.n_epoch = 400 #300
        args.epoch_per_test = 4
        
    elif args.dataset == 'WikiPeople-3':
        #args.lr = 0.00093 #GETD recommended
        #args.decay_rate = 0.995 #GETD recommended
        
#        args.lr = 0.007841793357665994
#        args.lamb = 0.15907392708072735
#        args.n_dim = 512  
#        args.decay_rate = 0.9929996778198179
#        args.n_batch = 1024
        
#        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.306
#        args.lr = 0.002560952212960611
#        args.input_dropout = 0.05282623057586833
#        args.hidden_dropout = 0.0536280780912419
#        args.n_batch = 1024
#        args.decay_rate = 0.9970654196449983
        
#        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.357
#        args.lr = 0.009199307620308197
#        args.input_dropout = 0.5501996937008095
#        args.hidden_dropout = 0.4521582172490322
#        args.n_batch = 512
#        args.decay_rate = 0.993131395203849    
#        args.n_dim = 512  
        
        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.342
        args.lr = 0.008602176931971298
        args.input_dropout = 0.6585684466626693
        args.hidden_dropout = 0.5093531687224002
        args.n_batch = 128
        args.decay_rate = 0.9910816415150104
        args.n_dim = 512  
        
        args.n_epoch = 600
        args.epoch_per_test = 3 #20
        
    elif args.dataset == 'JF17K-3':
        #args.lr = 0.00087
        #args.decay_rate = 0.99
                
#        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.644
#        args.lr = 2.9096546311298453e-05
#        args.input_dropout = 0.05878058892426788
#        args.hidden_dropout = 0.14927461694919153
#        args.n_batch = 128
#        args.decay_rate = 0.9938097929046755
        
#        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.670
#        args.lr = 0.0014502826047005277
#        args.input_dropout = 0.04239634993464285
#        args.hidden_dropout = 0.08890583577484336
#        args.n_batch = 128
#        args.decay_rate = 0.9911779527560267
        
#        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.672
#        args.lr = 0.00041222535646275624
#        args.input_dropout = 0.0006067135864501887
#        args.hidden_dropout = 0.10004344200961111
#        args.n_batch = 128
#        args.decay_rate = 0.9933335801000462
        
        
        # hyperOpt on ART under: adam, binary cross entropy loss, batch/drop, nCP0.678
        args.lr = 0.0001042347445944483
        args.input_dropout = 0.4622459903881589
        args.hidden_dropout = 0.4569806402605896
        args.n_batch = 128
        args.decay_rate = 0.9931621315787987
               
                       
        args.n_epoch = 600
        args.epoch_per_test = 3 #20
        
    return args

