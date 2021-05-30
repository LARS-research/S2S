import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])



"""
class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam([self.model.arch_parameters()],
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    #self.cluster_rela_dict = cluster_rela_dict

  def step(self, h, t, r, eta, network_optimizer):
    self.optimizer.zero_grad()
    self._backward_step(h, t, r, updateType="alphas")
    self.optimizer.step()
#    print("struct after loss backward")
#    print(self.model._arch_parameters.data.view(64,3)[0:10,:])

  def _backward_step(self, h, t, r, updateType):
    #print("binarization perfromed")
#    print("=====struct====")
#    print("struct before binarization")
#    print(self.model._arch_parameters.data.view(64,3)[0:10,:])
    self.model.binarization(tau_state=True)
#    print("struct after binarization")
#    print(self.model._arch_parameters.data.view(64,3)[0:10,:])
    loss = self.model._loss(h, t, r, updateType)
    loss += self.model.args.lamb * self.model.regul
    loss.backward()
#    print("struct after loss")
#    print(self.model._arch_parameters.data.view(64,3)[0:10,:])
    self.model.restore()
#    print("struct after restore")
#    print(self.model._arch_parameters.data.view(64,3)[0:10,:])
"""



class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam([self.model.arch_parameters()],
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)


  def step(self, h, t, r, eta, network_optimizer):
    self.optimizer.zero_grad()
    self._backward_step(h, t, r, updateType="alphas")
    self.optimizer.step()


  def _backward_step(self, h, t, r, updateType):

    self.model.binarization(tau_state=True)
    loss = self.model._loss(h, t, r, updateType)
    loss += self.model.args.lamb * self.model.regul
    loss.backward()

    self.model.restore()

    




