import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time


class KGEModule(nn.Module):
    def __init__(self, n_ent, n_rel, args, arch):
        super(KGEModule, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        
        self.op_idx = torch.LongTensor(arch)
        
        self.args = args
        self.n_dim = args.n_dim
        self.lamb = args.lamb
        
        self.ent_embed = nn.Embedding(n_ent, args.n_dim)
        self.rel_embed = nn.Embedding(n_rel, args.n_dim)
        self.init_weight()
        
        self.input_dropout = torch.nn.Dropout(self.args.input_dropout)
        self.hidden_dropout = torch.nn.Dropout(self.args.hidden_dropout)
        self.bne = torch.nn.BatchNorm1d(self.n_dim)
        self.bnr = torch.nn.BatchNorm1d(self.n_dim)
        self.bnw = torch.nn.BatchNorm1d(self.n_dim)
        
        self.K = args.num_blocks
        self.GPU = args.GPU
        
        self.n_arity = args.n_arity + 1
        self.tau = 0.5
        
                
        if self.GPU:
            self.ops = Variable(torch.Tensor([[0, 1, -1] for i in range(self.K**self.n_arity)]).cuda())
        else:
            self.ops = Variable(torch.Tensor([[0, 1, -1] for i in range(self.K**self.n_arity)]))
        


    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)


    """
    binarity
    """
    
    def forward(self, facts, arch):
        #self.cluster_rela_dict = cluster_rela_dict
        
        """convert the architect into struct list"""
        #max_idx = torch.LongTensor([item.index(True) for item in arch.tolist()])
        max_idx= arch
        
        head, tail, rela = facts[:,0], facts[:,1], facts[:,2]
        
        head = head.view(-1)
        tail = tail.view(-1)
        rela = rela.view(-1)
        
        length = self.n_dim // self.K  
        head_embed = self.input_dropout(self.bne(self.ent_embed(head))).view(-1, self.K, length)
        tail_embed = self.input_dropout(self.bne(self.ent_embed(tail))).view(-1, self.K, length)
        rela_embed = self.bnr(self.rel_embed(rela)).view(-1, self.K, length)
        
        head_scores1 = self.bin_neg_other(rela_embed, tail_embed, max_idx, 1)
        tail_scores1 = self.bin_neg_other(rela_embed, head_embed, max_idx, 2)
        
        head_scores = F.softmax(head_scores1, dim=1)
        tail_scores = F.softmax(tail_scores1, dim=1)
        
        size = head_scores.size()
                
        indices = torch.LongTensor([i for i in range(size[0])])
        head_label = torch.zeros(size)
        tail_label = torch.zeros(size)
        
        head_label[indices, head] = 1.
        tail_label[indices, tail] = 1.
                
        loss = F.binary_cross_entropy(head_scores, head_label) + \
               F.binary_cross_entropy(tail_scores, tail_label)
                
#        extend_r1 = self.get_r1(rela_embed, head_embed, max_idx, 2)        
#        pos_trip = self.test_trip(extend_r1, tail_embed)
#        neg_tail = self.bin_neg_last(extend_r1)
#        neg_head = self.bin_neg_other(rela_embed, tail_embed, max_idx, 1)
        
        
        
#        max_t = torch.max(neg_tail, 1, keepdim=True)[0]        
#        max_h = torch.max(neg_head, 1, keepdim=True)[0]
#
#        loss = - 2*pos_trip + max_t + torch.log(torch.sum(torch.exp(neg_tail - max_t), 1)) +\
#               max_h + torch.log(torch.sum(torch.exp(neg_head - max_h), 1))
#       self.regul = torch.sum(rela_embed**2)

        return loss
        
    def extend(self, embed):
        size = list(embed.size())
        size.insert(2,self.K)
        if self.GPU:
            extend_embed = torch.zeros(size).cuda()
        else:
            extend_embed = torch.zeros(size)
        for i in range(size[2]):
            extend_embed[:,:,i] = embed
        return extend_embed
    
    def inner_vec(self, extend_embed, embed):
        size = list(extend_embed.size())
        if self.GPU:
            vec = torch.zeros(extend_embed.size()).cuda()
        else:
            vec = torch.zeros(extend_embed.size())
        for i in range(size[1]):
            vec[:,i] = extend_embed[:,i] * embed
        return vec
    
    def test_trip(self, extend_r1, embed):
        score = torch.sum(extend_r1, 1) * embed 
        return torch.sum(torch.sum(score,2),1)
    
    def get_r1(self, r, e1, max_idx, last_idx):
        extend_r = self.extend(r)
        r1 = self.inner_vec(extend_r, e1).view(-1, self.K*self.K, self.n_dim//self.K)
        extend_r1 = self.extend(r1)
        extend_r1 = self.inner_alpha_bi(extend_r1, max_idx, last_idx)
        
        return extend_r1
    
    def inner_alpha_bi(self, extend_r1, op_idx, last_idx):
        core_size = [self.K for i in range(self.n_arity)]
        cube_indices = torch.LongTensor([i for i in range(self.K**self.n_arity)])
        
        alpha = self.ops[cube_indices, op_idx].view(core_size)
        
        for i in range(self.K):
            if last_idx == 2:
                extend_r1[:,:,i] = extend_r1[:,:,i] * alpha[:,:,i].reshape(1, self.K*self.K, 1)
            elif last_idx == 1:
                extend_r1[:,:,i] = extend_r1[:,:,i] * alpha[:,i,:].reshape(1, self.K*self.K, 1)
        
        return extend_r1
    
    def bin_neg_last(self, extend_re):
        vec_re = torch.sum(extend_re,1).view(-1, self.n_dim)
        
        e_embed = self.ent_embed.weight
        scores = torch.mm(vec_re, e_embed.transpose(1,0))
        return scores
    
    
    def bin_neg_other(self, rela, e, op_idx, last_idx): 
        
        extend_re = self.get_r1(rela, e,  op_idx, last_idx)
        vec_re = torch.sum(extend_re,1).view(-1, self.n_dim)
        
        e_embed = self.ent_embed.weight
        scores = torch.mm(vec_re, e_embed.transpose(1,0))
        return scores

           
    def get_r12(self, r_embed, e1_embed, e2_embed, max_idx, last_idx):
        extend_r = self.extend(r_embed)
        r1 = self.inner_vec(extend_r, e1_embed).view(-1, self.K**2, self.n_dim//self.K)
        
        extend_r1 = self.extend(r1)
        r12 = self.inner_vec(extend_r1, e2_embed).view(-1, self.K**3, self.n_dim//self.K)
        
        extend_r12 = self.extend(r12)
        extend_r12 = self.inner_alpha_tri(extend_r12, max_idx, last_idx)

        return extend_r12
    
    def inner_alpha_tri(self, extend_r12, op_idx, last_idx):
        core_size = [self.K for i in range(self.n_arity)]
        cube_indices = torch.LongTensor([i for i in range(self.K**self.n_arity)])
        
        alpha = self.ops[cube_indices, op_idx].view(core_size)
        
        for i in range(self.K):
            if last_idx == 3:
                extend_r12[:,:,i] = extend_r12[:,:,i] * alpha[:,:,:,i].reshape(1, self.K**3, 1)
            elif last_idx == 2:
                extend_r12[:,:,i] = extend_r12[:,:,i] * alpha[:,:,i,:].reshape(1, self.K**3, 1)
            elif last_idx == 1:
                extend_r12[:,:,i] = extend_r12[:,:,i] * alpha[:,i,:,:].reshape(1, self.K**3, 1)
                
        return extend_r12

    def tri_neg_other(self, rela, e1, e2, op_idx, last_idx):
        extend_r12 = self.get_r12(rela, e1, e2, op_idx, last_idx)
        vec_r12 = torch.sum(extend_r12,1).view(-1, self.n_dim)
        
        # hidden dropout and batch normalization
        if self.args.loss == "binCrossEntropy":
            #print("tri_neg_other yes")
            vec_r12 = self.hidden_dropout(self.bnw(vec_r12))
        elif self.args.loss == "extend":
            vec_r12 = self.hidden_dropout(vec_r12)
            #vec_r12 = self.hidden_dropout(self.bnw(vec_r12))
        
        e_embed = self.ent_embed.weight
        scores = torch.mm(vec_r12, e_embed.transpose(1,0))
        return scores
    
    def tri_neg_last(self, extend_r12):
        vec_r12 = torch.sum(extend_r12,1).view(-1, self.n_dim)
        
        # hidden dropout and batch normalization
        if self.args.loss == "binCrossEntropy":
            #print("tri_neg_last yes")
            vec_r12 = self.hidden_dropout(self.bnw(vec_r12))
        elif self.args.loss == "extend":
            vec_r12 = self.hidden_dropout(vec_r12)
            #vec_r12 = self.hidden_dropout(self.bnw(vec_r12))
        
        e_embed = self.ent_embed.weight
        scores = torch.mm(vec_r12, e_embed.transpose(1,0))
        return scores

    
    """
    3-arity
    """
    def forward_tri(self, facts, arch):
        
        """convert the architect into struct list"""
        #max_idx = torch.LongTensor([item.index(True) for item in arch.tolist()])
        max_idx = arch
        
        r, e1, e2, e3 = facts[:,0], facts[:,1], facts[:,2], facts[:,3]
        r, e1, e2, e3 = r.view(-1), e1.view(-1), e2.view(-1), e3.view(-1)
        
        
        if self.args.loss == "binCrossEntropy":
            
            # dropout
            length = self.n_dim // self.K  
            r_embed = self.bnr(self.rel_embed(r)).view(-1, self.K, length)
            e1_embed = self.input_dropout(self.bne(self.ent_embed(e1))).view(-1, self.K, length)
            e2_embed = self.input_dropout(self.bne(self.ent_embed(e2))).view(-1, self.K, length)
            e3_embed = self.input_dropout(self.bne(self.ent_embed(e3))).view(-1, self.K, length)

            e1_scores = F.softmax(self.tri_neg_other(r_embed, e2_embed, e3_embed, max_idx, 1), dim=1)
            e2_scores = F.softmax(self.tri_neg_other(r_embed, e1_embed, e3_embed, max_idx, 2), dim=1)
            e3_scores = F.softmax(self.tri_neg_other(r_embed, e1_embed, e2_embed, max_idx, 3), dim=1)
                    
            size = e1_scores.size()
            indices = torch.LongTensor([i for i in range(size[0])])
            
            if self.GPU:
                e1_label = torch.zeros(size).cuda()
                e2_label = torch.zeros(size).cuda()
                e3_label = torch.zeros(size).cuda()
            else:
                e1_label = torch.zeros(size)
                e2_label = torch.zeros(size)
                e3_label = torch.zeros(size)
                
            e1_label[indices, e1] = 1.
            e2_label[indices, e2] = 1.
            e3_label[indices, e3] = 1.
            
            loss = F.binary_cross_entropy(e1_scores, e1_label) + \
                   F.binary_cross_entropy(e2_scores, e2_label) + \
                   F.binary_cross_entropy(e3_scores, e3_label)
        
        elif self.args.loss == "extend":
            length = self.n_dim // self.K  
            r_embed = self.rel_embed(r).view(-1, self.K, length)
            e1_embed = self.input_dropout(self.ent_embed(e1)).view(-1, self.K, length)
            e2_embed = self.input_dropout(self.ent_embed(e2)).view(-1, self.K, length)
            e3_embed = self.input_dropout(self.ent_embed(e3)).view(-1, self.K, length)
            

            extend_r12 = self.get_r12(r_embed, e1_embed, e2_embed, max_idx, 3)
                
            pos_trip = self.test_trip(extend_r12, e3_embed)
            
            neg_e3 = F.softmax(self.tri_neg_last(extend_r12), dim=1) 
            neg_e2 = F.softmax(self.tri_neg_other(r_embed, e1_embed, e3_embed, max_idx, 2), dim=1) 
            neg_e1 = F.softmax(self.tri_neg_other(r_embed, e2_embed, e3_embed, max_idx, 1), dim=1)
                    
            max_neg_e1 = torch.max(neg_e1, 1, keepdim=True)[0]
            max_neg_e2 = torch.max(neg_e2, 1, keepdim=True)[0]
            max_neg_e3 = torch.max(neg_e3, 1, keepdim=True)[0]
            
            loss1 = - 3*pos_trip + max_neg_e1 + torch.log(torch.sum(torch.exp(neg_e1 - max_neg_e1), 1)) +\
                   max_neg_e2 + torch.log(torch.sum(torch.exp(neg_e2 - max_neg_e2), 1)) +\
                   max_neg_e3 + torch.log(torch.sum(torch.exp(neg_e3 - max_neg_e3), 1))
                   
            self.regul = torch.sum(r_embed**2)
            
            loss = torch.sum(loss1) + self.args.lamb * self.regul
                
        return loss
    
    
    
    def _loss(self, facts, arch):
        if self.n_arity == 3:
            return self.forward(facts, arch)
        elif self.n_arity == 4:
            return self.forward_tri(facts, arch)



    """
    def test_trip_fast(self, r_embed, e1_embed, e2_embed, op_idx):
        n_head = r_embed.size(0)
        r1 = torch.einsum('bij,bmj->bimj', r_embed, e1_embed).view(n_head, self.K**2, 128)
        r12 = torch.einsum('bij,bmj->bimj', r1, e2_embed) * op_idx.view(1, self.K**2, self.K, 1)
        score = torch.sum(r12, 1)
        return torch.sum(score.view(-1, self.n_dim), 1)
        
    
    
    
    def forward_fast(self, facts, arch):
        
                
        op_idx = torch.LongTensor([item.index(True) for item in arch.tolist()])
        
        r, e1, e2 = facts[:,0], facts[:,1], facts[:,2]
        r, e1, e2 = r.view(-1), e1.view(-1), e2.view(-1)
        
        self.length = self.n_dim//self.K
        r_embed = self.rel_embed(r).view(-1, self.K, self.length)
        e1_embed = self.ent_embed(e1).view(-1, self.K, self.length)
        e2_embed = self.ent_embed(e2).view(-1, self.K, self.length)
        
        pos_trip = self.test_trip_fast(r_embed, e1_embed, e2_embed, op_idx)
        
        neg_tail = self.bin_neg_last(extend_r1)
        
    """

    




