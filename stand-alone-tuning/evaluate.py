import os
import torch
import argparse
import numpy as np
from read_data import nary_dataloader
from read_data_autosf import DataLoader, n_ary_heads
from corrupter import BernCorrupter
from utils import logger_init, plot_config, gen_struct, default_search_hyper
from select_gpu import select_gpu
from base_model import BaseModel
from collections import defaultdict
from hyperopt_master.hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial


"""
Build Default Arguments
"""
def register_default_args():
    
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    
    # dataset dir
    #parser.add_argument('--task_dir', type=str, default='../KG_Data/FB15K237', help='the directory to dataset')
    parser.add_argument('--task_dir', type=str, default='/export/data/sdiaa/KG_Data/WN18RR', help='the directory to dataset')
    
    #parser.add_argument('--gpu', type=int, default=4, help='set gpu #')
    parser.add_argument('--parrel', type=int, default=1, help='set gpu #')
    parser.add_argument('--lr', type=float, default=0.07, help='set learning rate')
    parser.add_argument('--n_epoch', type=int, default=300, help='number of training epochs')
    parser.add_argument('--n_shared_epoch', type=int, default=500, help='')
    parser.add_argument('--n_controller_epoch', type=int, default=20, help='step for controller parameters')
    parser.add_argument('--n_derive_sample', type=int, default=1, help='')
    parser.add_argument('--test_batch_size', type=int, default=512, help='')
    
    # hyper-parameters for nn training
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--model', type=str, default='random', help='model type')
    parser.add_argument('--save', type=bool, default=False, help='whether save model')
    parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
    
    #parser.add_argument('--optim', type=str, default='adagrad', help='optimization method')
    parser.add_argument('--optim', type=str, default='adam', help='optimization method')
    
    parser.add_argument('--lamb', type=float, default=0.4, help='set weight decay value')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='set weight decay value')
    parser.add_argument('--n_dim', type=int, default=512, help='set embedding dimension')
    parser.add_argument('--n_sample', type=int, default=25, help='number of negative samples')
    parser.add_argument('--classification', type=bool, default=False, help='number of negative samples')
    parser.add_argument('--cmpl', type=bool, default=False, help='whether use complex value or not')
    
    parser.add_argument('--n_batch', type=int, default=4096, help='number of training batches')
    
    parser.add_argument('--epoch_per_test', type=int, default=10, help='frequency of testing')
    parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
    parser.add_argument('--out_file_info', type=str, default='_tune', help='extra string for the output file name')
    parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
    parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')
    parser.add_argument('--log_prefix', type=str, default='', help='log prefix')
    
    #args = parser.parse_args()
    
    return parser


def get_data_idxs(data, entity_idxs, relation_idxs):
    if len(data[0])-1 == 3:
        data_idxs = [(relation_idxs[data[i][0]], entity_idxs[data[i][1]], entity_idxs[data[i][2]], entity_idxs[data[i][3]]) for i in range(len(data))]
    elif len(data[0])-1 == 4:
        data_idxs = [(relation_idxs[data[i][0]], entity_idxs[data[i][1]], entity_idxs[data[i][2]], entity_idxs[data[i][3]], entity_idxs[data[i][4]]) for i in range(len(data))]
    return data_idxs

"""
main function
"""
def main(args, arch):
#def main(args, i):

    # set number of threads in pytorch
    torch.set_num_threads(6)
    
    # select which gpu to use
    logger_init(args)
    
    # set gpu
    if args.GPU:
        torch.cuda.set_device(args.gpu) 
        
    # the default settings for correspdonding dataset
    args = default_search_hyper(args)
    
#    hyperOpt = {"lr":[0.00635456700742798, 0.0049700352658686425, 0.0023726642982752643],
#                "lamb":[3.162503061522238e-05, 1.9567149674424395e-05, 1.0729611255307008e-05],
#                "d":[512, 512, 512],
#                "dr":[0.9933500551931267, 0.9903909316509071, 0.9933910046627364],
#                "batch_size":[256, 256, 256]}
#    
#    args.lr = hyperOpt["lr"][i]
#    args.lamb = hyperOpt["lamb"][i]
#    args.n_dim = hyperOpt["d"][i]
#    args.decay_rate = hyperOpt["dr"][i]
#    args.n_batch = hyperOpt["batch_size"][i]


    # load data
    # read nary data
    if args.n_arity > 2:
        d = nary_dataloader(args.task_dir)
        
        entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        n_ent, n_rel = len(entity_idxs), len(relation_idxs)
        print("Number of train:{}, valid:{}, test:{}.".format(len(d.train_data), len(d.valid_data), len(d.test_data)))
        
        train_data = torch.LongTensor(get_data_idxs(d.train_data, entity_idxs, relation_idxs))
        valid_data = torch.LongTensor(get_data_idxs(d.valid_data, entity_idxs, relation_idxs))
        test_data = torch.LongTensor(get_data_idxs(d.test_data, entity_idxs, relation_idxs))

        e1_sp, e2_sp, e3_sp = n_ary_heads(train_data, valid_data, test_data)
        
#        train_data = torch.LongTensor(get_data_idxs(d.train_data, entity_idxs, relation_idxs))[0:512]
#        valid_data = torch.LongTensor(get_data_idxs(d.valid_data, entity_idxs, relation_idxs))[0:512]
#        test_data = torch.LongTensor(get_data_idxs(d.test_data, entity_idxs, relation_idxs))[0:512]
        
    else:
        loader = DataLoader(args.task_dir)
        n_ent, n_rel = loader.graph_size()
        train_data = loader.load_data('train')
        valid_data = loader.load_data('valid')
        test_data  = loader.load_data('test')
        print("Number of train:{}, valid:{}, test:{}.".format(len(train_data[0]), len(valid_data[0]), len(test_data[0])))
    
        heads, tails = loader.heads_tails()
            
        train_data = torch.LongTensor(train_data).transpose(0,1)#[0:512]
        valid_data = torch.LongTensor(valid_data).transpose(0,1)#[0:512]
        test_data = torch.LongTensor(test_data).transpose(0,1)#[0:512]
    
    file_path = "fix_nary" + "_" + str(args.num_blocks)
    directory = os.path.join("results", args.dataset, file_path)
    args.out_dir = directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.environ["OMP_NUM_THREADS"] = "4"   
    os.environ["MKL_NUM_THREADS"] = "4"   
    args.perf_file = os.path.join(directory, args.dataset + '_fix_nary_' + str(args.num_blocks) + "_" + str(args.trial) + '.txt')

    print('output file name:', args.perf_file)
    
    plot_config(args)
    
    def tester_val(facts = None, arch = None):
        if args.n_arity == 2:       
            return model.test_link(test_data=valid_data, n_ent=n_ent, heads=heads, tails=tails, filt=args.filter, arch=arch)
            
        elif args.n_arity > 2:
            return model.evaluate(valid_data, e1_sp, e2_sp, e3_sp, arch)
    
    def tester_tst():
        if args.n_arity == 2:  
            return model.test_link(test_data=test_data, n_ent=n_ent, heads=heads, tails=tails, filt=args.filter)
        elif args.n_arity > 2:
            return model.evaluate(test_data, e1_sp, e2_sp, e3_sp)
    
    tester_trip_class = None
    model = BaseModel(n_ent, n_rel, args, arch)   
    model.train(train_data, valid_data, tester_val, tester_tst, tester_trip_class)
    

if __name__ == '__main__':
    
    parser = register_default_args()
    
    parser.add_argument('--loss', type=str, default="binCrossEntropy", help='extend or binCrossEntropy')
    
    parser.add_argument('--input_dropout', type=float, default=0.3740, help='')
    parser.add_argument('--hidden_dropout', type=float, default=0.4513, help='')
    
    parser.add_argument('--num_blocks', type=int, default=4, help='')
    parser.add_argument('--dataset', type=str, default="JF17K-3", help='')
    parser.add_argument('--n_arity', type=int, default=3, help='')
#    parser.add_argument('--dataset', type=str, default="WN18RR", help='')
#    parser.add_argument('--n_arity', type=int, default=2, help='')
    parser.add_argument('--GPU', type=bool, default=True, help='')
    parser.add_argument('--gpu', type=int, default=3, help='set gpu #')                    
    parser.add_argument('--trial', type=str, default="ART_Hyper_adam_bin_batch_drop_nCP0.678", help='')

    args = parser.parse_args()
    
    if args.n_arity == 2:   
        args.task_dir = "../KG_Data_binary/" + args.dataset
    else:
        args.task_dir = "../KG_Data_nary/" + args.dataset

        
    if args.dataset == "WikiPeople-3":
        arch_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    elif args.dataset == "JF17K-3":
        arch_list = [[1, 1, 0, 2, 1, 0, 2, 1, 2, 2, 1, 0, 2, 0, 1, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 2, 0, 1, 0, 2, 1, 1, 1, 2, 1, 0, 2, 0, 2, 2, 1, 1, 2, 2, 0, 1, 0, 2, 1, 0, 2, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 1, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 2, 2, 1, 2, 1, 0, 2, 1, 2, 0, 2, 2, 2, 1, 1, 0, 0, 2, 0, 1, 2, 1, 2, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 0, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 1, 2, 2, 0, 1, 1, 0, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 2, 0, 0, 2, 2, 1, 1, 1, 0, 1, 1, 2, 1, 2, 0, 0, 2, 0, 0, 2, 1, 0, 1, 0, 1, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0, 2, 0, 2, 1, 2, 2, 1, 0, 1]]
    
    #arch_list = [[1, 1, 1, 1, 2, 2, 0, 2, 2, 1, 1, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 1, 1, 2, 2, 0, 2, 2, 2, 2, 0, 2, 1, 2, 1, 0, 2, 0, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 0, 1, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 1, 0, 1, 0, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 0, 2, 0, 1, 2, 2, 1, 1, 2, 1, 2, 0, 0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 0, 1, 0, 2, 0, 0, 1, 1, 2, 1, 2, 2, 1, 1, 0, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 0, 2, 2, 1, 2, 1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 2, 2, 1, 2, 0, 1, 2, 2, 1, 1, 2, 0, 2, 2, 1, 0, 0, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 0, 0, 1, 1, 2, 0, 1, 0, 1, 2, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 2, 1, 2, 0, 2, 1, 1, 2]]
            
    for arch in arch_list:       
        main(args, arch)
        
        
    
    
    
    

    
    
                



