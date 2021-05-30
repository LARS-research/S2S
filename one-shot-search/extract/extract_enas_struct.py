#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:43:28 2020

@author: shimin
"""

import numpy as np
import ast



def topK(filePath, outfilePath, K, rela_cluster):
    f = open(filePath)
    l = []
    for line in f:
        if len(line.split(":")) == 2:
            x = line.split(":")[1]
            x = ast.literal_eval(x)
            l.append(x)
            
    if rela_cluster == "scu":
        rewards, structs, relas = np.asarray(l[0]), np.asarray(l[1]), np.asarray(l[2])
    elif rela_cluster == "pde":
        rewards, structs = np.asarray(l[0]), np.asarray(l[1])
    
    topK_indices = rewards.argsort()[-K:][::-1]
    
    #print(rewards[topK_indices])
    
    
    with open(outfilePath, "a+") as f:
        for i in range(K):
            outstr_reward = "MRR:" + str(rewards[topK_indices][i]) + "\n"
            outstr_struct = "struct:" + str(list(structs[topK_indices][i].tolist())) + "\n"
            
            if rela_cluster == "scu":
                outstr_rela = "Rela:" + str(list(relas[topK_indices][i].tolist())) + "\n"
            elif rela_cluster == "pde":
                outstr_rela = "Rela:" + rela_cluster + "\n"
            
            
            f.write(outstr_reward)
            f.write(outstr_rela)
            f.write(outstr_struct)
            f.write("\n")


def bomK(filePath, outfilePath, K, rela_cluster):
    
    f = open(filePath)
    l = []
    for line in f:
        if len(line.split(":")) == 2:
            x = line.split(":")[1]
            x = ast.literal_eval(x)
            l.append(x)
    
    #print(l)
    if rela_cluster == "scu":
        rewards, structs, relas = np.asarray(l[0]), np.asarray(l[1]), np.asarray(l[2])
    elif rela_cluster == "pde":
        rewards, structs = np.asarray(l[0]), np.asarray(l[1])
        
    
    #topK_indices = rewards.argsort()[-K:][::-1]
    #print(rewards[topK_indices])
    #print(rewards[-K:])
    
    with open(outfilePath, "a+") as f:
        for i in range(K):
            outstr_reward = "MRR:" + str(rewards[-K:][i]) + "\n"
            outstr_struct = "struct:" + str(list(structs[-K:][i].tolist())) + "\n"
            
            if rela_cluster == "scu":
                outstr_rela = "Rela:" + str(list(relas[-K:][i].tolist())) + "\n"
            elif rela_cluster == "pde":
                outstr_rela = "Rela:" + rela_cluster + "\n"
            
            
            f.write(outstr_reward)
            f.write(outstr_rela)
            f.write(outstr_struct)
            f.write("\n")




filePath = "results/FB15K237/oas_scu_4_6/FB15K237_oas_scu_4_6_1.txt"

outfile_top_Path = "results/FB15K237/oas_scu_4_6/FB15K237_oas_scu_4_6_1_top_struct.txt"
outfile_bottom_Path = "results/FB15K237/oas_scu_4_6/FB15K237_oas_scu_4_6_1_bot_struct.txt"

K = 50

rela_cluster = "scu"

topK(filePath, outfile_top_Path, K, rela_cluster)
bomK(filePath, outfile_bottom_Path, K, rela_cluster)









