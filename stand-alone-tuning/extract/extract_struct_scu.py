#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:43:40 2020

@author: shimin
"""

import os

dirPath = "results/FB15K/rc_scu_-1_4_4/1"


files = os.listdir(dirPath)

predix = "_".join(files[0].split("_")[:3])

fileName_final = predix + "_final.txt"
fileName_finalTau =  predix + "_final_tau.txt"
fileName_best = predix + "_best.txt"

filePath_final = os.path.join(dirPath, fileName_final)
filePath_finalTau = os.path.join(dirPath, fileName_finalTau)
filePath_best = os.path.join(dirPath, fileName_best)

try:
    files.remove(".DS_Store")
except:
    print("nothing")
    


taus = []
for fileName in files:
    taus.append(float(fileName.split("_")[3]))
taus = sorted(taus)

files = []
for tau in taus:
    file = predix + "_" + str(tau) + "_.txt"
    files.append(file)

for fileName in files:
    filePath = os.path.join(dirPath, fileName)
    with open(filePath) as f:
        count = 0
        for line in f:
            
            if len(line.split(":")) == 2 and line.split(":")[0] == "rela_cluster":
                with open(filePath_final, "a") as f:
                    f.write(line+"\n")
                with open(filePath_finalTau, "a") as f:
                    f.write(line+"\n")
                with open(filePath_best, "a") as f:
                    f.write(line+"\n")
                
            
            elif len(line.split(":")) == 2 and count < 2:
                count += 1
                with open(filePath_final, "a") as f:
                    f.write(line)
                with open(filePath_finalTau, "a") as f:
                    f.write(line)
                with open(filePath_best, "a") as f:
                    f.write(line)
                                
            elif len(line.split(":")) == 2 and count == 2:
                count += 1
                with open(filePath_final, "a") as f:
                    f.write(line)
                
                
            elif len(line.split(":")) == 2 and count == 3:
                count += 1
                with open(filePath_best, "a") as f:
                    f.write(line)
                
            elif len(line.split(":")) == 2 and count == 4:
                count += 1
                with open(filePath_finalTau, "a") as f:
                    f.write(line)
            
    
            
    
            
    
        
        