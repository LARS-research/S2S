import torch
import torch.nn as nn
import numpy as np
from metrics import mrr_mr_hitk
from utils import batch_by_size
import logging
import time
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ExponentialLR
from models import KGEModule
from architect import Architect
from asng.categorical_asng import CategoricalASNG
from collections import defaultdict

import torch.nn.functional as F



class BaseModel(object):
    def __init__(self, n_ent, n_rel, args):
        
        self.model = KGEModule(n_ent, n_rel, args)
        
        if args.GPU:
            self.model.cuda()
            
        self.n_ent = n_ent
        self.n_rel = n_rel
        
        self.time_tot = 0
        self.args = args
        self.n_dim = args.n_dim
        self.K = args.num_blocks
        self.GPU = args.GPU
        
        self.n_arity = args.n_arity + 1
        
        # initialize the arch parameters
        self.n_ops = self.K**self.n_arity
        self.categories = np.asarray([3 for i in range(self.n_ops)])
        alpha, init_delta, trained_theta = 1.5, 1.0, None
        self.asng = CategoricalASNG(self.categories, alpha=alpha, init_delta=init_delta, init_theta=trained_theta)
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))

    def get_reward(self, facts):
        
        if self.args.M_val == "loss":
            archs, loss_archs = [],[]
            with torch.no_grad():
                for i in range(2):
                    arch = self.asng.sampling()
                    archs.append(arch)
                    
                    loss_arch = self.model._loss(facts, arch)
                    #loss_arch += self.model.args.lamb * self.model.regul
                    loss_archs.append(loss_arch)
        
        elif self.args.M_val == "mrr":
            archs, loss_archs = [], []
            with torch.no_grad():
                for i in range(2):
                    arch = self.asng.sampling()
                    archs.append(arch)
                    
                    result = self.tester_val(facts, arch)
#                    result = self.tester_val(arch = arch)
                    loss_archs.append(-result[0])
                    
                    
#                    if result[0] > 0.3200:
#                        self.good_struct.append([item.index(True) for item in arch.tolist()])
#                    if result[0] > self.best_mrr:
#                        self.best_mrr = result[0]
#                        self.best_struct = [item.index(True) for item in arch.tolist()]
                    
        return archs, loss_archs#, embed_time, mrr_time


    def train(self, train_data, valid_data, tester_val, tester_tst, tester_trip=None):
        
        self.tester_val = tester_val
        
        if self.args.optim=='adam' or self.args.optim=='Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim=='adagrad' or self.args.optim=='Adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)

        scheduler = ExponentialLR(self.optimizer, self.args.decay_rate)

        n_epoch = self.args.n_epoch
        n_batch = self.args.n_batch
        self.best_mrr = 0
        
        # useful information related to cache
        n_train = train_data.size(0)
        n_valid = valid_data.size(0)
        
        self.good_struct = []
        
        for epoch in range(n_epoch):

            self.model.train()
            
            start = time.time()

            self.epoch = epoch
            rand_idx = torch.randperm(n_train)
            
            if self.GPU:
                train_data = train_data[rand_idx].cuda()
            else:
                train_data = train_data[rand_idx]
            
            epoch_loss = 0
            
            for facts in batch_by_size(n_batch, train_data, n_sample=n_train):
                
                if self.args.valid_batch > 0:
                    """only activated when valid for architecture update needed"""
                    rand_idx_valid = torch.randperm(n_valid)[0: self.args.valid_batch]
                    if self.GPU:
                        data_v = valid_data[rand_idx_valid].cuda()
                    else:
                        data_v = valid_data[rand_idx_valid]

                    archs, loss_archs = self.get_reward(data_v)
                    self.asng.update(np.asarray(archs), np.asarray(loss_archs), range_restriction=True)

                else:
                    archs, loss_archs = self.get_reward(facts)
                    self.asng.update(np.asarray(archs), np.asarray(loss_archs), range_restriction=True)
                    
                
                arch = self.asng.sampling()
                
                self.model.zero_grad()
                if self.n_arity == 3:
                    loss = self.model.forward(facts, arch)       
                    loss.backward()
                    
                elif self.n_arity == 4:
                    loss = self.model.forward_tri(facts, arch)                
                    loss.backward()



                """kge step"""
                self.optimizer.step()
                self.prox_operator()
                
                epoch_loss += loss.data.cpu().numpy()            

            scheduler.step()
            self.time_tot += time.time() - start
            print("Epoch: %d/%d, Loss=%.8f, Time=%.4f"%(epoch+1, n_epoch, epoch_loss/n_train, time.time()-start))
            

            if (epoch+1) % self.args.epoch_per_test == 0:
                
                    valid_mrr, valid_mr, valid_1, valid_3, valid_10 = tester_val()
                    test_mrr,  test_mr,  test_1,  test_3,  test_10 =  tester_tst()
                    if tester_trip is None:
                        out_str = '%d\t%.2f %.2f \t%.4f  %.1f %.4f %.4f %.4f\t%.4f %.1f %.4f %.4f %.4f\n' % (epoch, self.time_tot, epoch_loss/n_train, \
                            valid_mrr, valid_mr, valid_1, valid_3, valid_10, \
                            test_mrr, test_mr, test_1, test_3, test_10)

                    with open(self.args.perf_file, 'a') as f:
                        f.write(out_str)

                    if test_mrr > self.best_mrr:
                        self.best_mrr = test_mrr
                        self.best_struct = torch.Tensor(self.asng.p_model.theta).argmax(1).tolist()
                    
        with open(self.args.perf_file, 'a') as f:
            
            final_struct = torch.Tensor(self.asng.p_model.theta).argmax(1).tolist()
            f.write("final arch:"+str(final_struct)+","+str(test_mrr)+"\n")
            if final_struct != self.best_struct:
                f.write("best arch:"+str(self.best_struct)+","+str(self.best_mrr)+"\n")
                
                
            if self.good_struct != []:
                for item in self.good_struct:
                    f.write("good arch:"+str(self.item)+"\n")
                
        return self.best_mrr

    def prox_operator(self,):
        for n, p in self.model.named_parameters():
            if 'ent' in n:
                X = p.data.clone()
                Z = torch.norm(X, p=2, dim=1, keepdim=True)
                Z[Z<1] = 1
                X = X/Z
                p.data.copy_(X.view(self.n_ent, -1))
    
    
    def name(self, idx):
        i = idx[0]
        i_rc =  self.rela_cluster[i]
        self.r_embed[i,:,:] = self.rel_embed_2K_1[i,self.idx_list[i_rc],:] * self.model._arch_parameters[i_rc][[j for j in range(self.K*self.K)], self.idx_list[i_rc]].view(-1,1)


    def test_link(self, test_data, n_ent, heads, tails, filt=True, arch=None):
        mrr_tot = 0.
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
                
        if arch is None:
            max_idx = torch.Tensor(self.asng.p_model.theta).argmax(1)
        else:
            max_idx = torch.LongTensor([item.index(True) for item in arch.tolist()])
        
        for facts in batch_by_size(self.args.test_batch_size, test_data):
            
            if self.GPU:
                                
                batch_h = facts[:,0].cuda()
                batch_t = facts[:,1].cuda()
                batch_r = facts[:,2].cuda()
            else:
                batch_h = facts[:,0]
                batch_t = facts[:,1]
                batch_r = facts[:,2]
                
            length = self.n_dim // self.K
            h_embed = self.model.ent_embed(batch_h).view(-1, self.K, length)
            t_embed = self.model.ent_embed(batch_t).view(-1, self.K, length)
            r_embed = self.model.rel_embed(batch_r).view(-1, self.K, length)
            
            head_scores = torch.sigmoid(self.model.bin_neg_other(r_embed, t_embed, max_idx, 1)).data
            tail_scores = torch.sigmoid(self.model.bin_neg_other(r_embed, h_embed, max_idx, 2)).data
            
            for h, t, r, head_score, tail_score in zip(batch_h, batch_t, batch_r, head_scores, tail_scores):
                h_idx = int(h.data.cpu().numpy())
                t_idx = int(t.data.cpu().numpy())
                r_idx = int(r.data.cpu().numpy())
                if filt:            # filter
                    if tails[(h_idx,r_idx)]._nnz() > 1:
                        tmp = tail_score[t_idx].data.cpu().numpy()
                        idx = tails[(h_idx, r_idx)]._indices()
                        tail_score[idx] = 0.0
                        
                        if self.GPU:
                            tail_score[t_idx] = torch.from_numpy(tmp).cuda()
                        else:
                            tail_score[t_idx] = torch.from_numpy(tmp)
                            
                    if heads[(t_idx, r_idx)]._nnz() > 1:
                        tmp = head_score[h_idx].data.cpu().numpy()
                        idx = heads[(t_idx, r_idx)]._indices()
                        head_score[idx] = 0.0
                        if self.GPU:
                            head_score[h_idx] = torch.from_numpy(tmp).cuda()
                        else:
                            head_score[h_idx] = torch.from_numpy(tmp)
                            
                mrr, mr, hit = mrr_mr_hitk(tail_score, t_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                mrr, mr, hit = mrr_mr_hitk(head_score, h_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                count += 2
        
        if arch is None:                
            logging.info('Test_MRR=%f, Test_MR=%f, Test_H=%f %f %f, Count=%d', float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count, count)
        
        return float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count #, total_loss/n_test

        
    def evaluate(self, test_data, e1_sp, e2_sp, e3_sp, arch = None):
        
        mrr_tot = 0.
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        
        self.model.eval()
        
        if arch is None:
            max_idx = torch.Tensor(self.asng.p_model.theta).argmax(1)
        else:
            max_idx = torch.LongTensor([item.index(True) for item in arch.tolist()])
            
            
        for facts in batch_by_size(self.args.test_batch_size, test_data):
            
            if self.GPU:
                r, e1, e2, e3 = facts[:,0].cuda(), facts[:,1].cuda(), facts[:,2].cuda(), facts[:,3].cuda()
            else:
                r, e1, e2, e3 = facts[:,0], facts[:,1], facts[:,2], facts[:,3]
                                
            length = self.n_dim // self.K
            
            
#            r_embed = self.model.rel_embed(r).view(-1, self.K, length)
#            e1_embed = self.model.ent_embed(e1).view(-1, self.K, length)
#            e2_embed = self.model.ent_embed(e2).view(-1, self.K, length)
#            e3_embed = self.model.ent_embed(e3).view(-1, self.K, length)
            
            r_embed = self.model.bnr(self.model.rel_embed(r)).view(-1, self.K, length)
            e1_embed = self.model.input_dropout(self.model.bne(self.model.ent_embed(e1))).view(-1, self.K, length)
            e2_embed = self.model.input_dropout(self.model.bne(self.model.ent_embed(e2))).view(-1, self.K, length)
            e3_embed = self.model.input_dropout(self.model.bne(self.model.ent_embed(e3))).view(-1, self.K, length)
            
            e1_scores = F.softmax(self.model.tri_neg_other(r_embed, e2_embed, e3_embed, max_idx, 1), dim=1).data
            e2_scores = F.softmax(self.model.tri_neg_other(r_embed, e1_embed, e3_embed, max_idx, 2), dim=1).data
            e3_scores = F.softmax(self.model.tri_neg_other(r_embed, e1_embed, e2_embed, max_idx, 3), dim=1).data
            
            for idx in range(len(r)):                
                r_idx, e1_idx, e2_idx, e3_idx = int(r[idx].data.cpu().numpy()), int(e1[idx].data.cpu().numpy()), int(e2[idx].data.cpu().numpy()), int(e3[idx].data.cpu().numpy())
                
                if e1_sp[(r_idx, e2_idx, e3_idx)]._nnz() > 1:
                    tmp = e1_scores[idx][e1_idx].data.cpu().numpy()
                    indic = e1_sp[(r_idx, e2_idx, e3_idx)]._indices()
                    e1_scores[idx][indic] = 0.0
                    if self.GPU:
                        e1_scores[idx][e1_idx] = torch.from_numpy(tmp).cuda()
                    else:
                        e1_scores[idx][e1_idx] = torch.from_numpy(tmp)
                        
                if e2_sp[(r_idx, e1_idx, e3_idx)]._nnz() > 1:
                    tmp = e2_scores[idx][e2_idx].data.cpu().numpy()
                    indic = e2_sp[(r_idx, e1_idx, e3_idx)]._indices()
                    e2_scores[idx][indic] = 0.0
                    if self.GPU:
                        e2_scores[idx][e2_idx] = torch.from_numpy(tmp).cuda()
                    else:
                        e2_scores[idx][e2_idx] = torch.from_numpy(tmp)
                    

                if e3_sp[(r_idx, e1_idx, e2_idx)]._nnz() > 1:
                    tmp = e3_scores[idx][e3_idx].data.cpu().numpy()
                    indic = e3_sp[(r_idx, e1_idx, e2_idx)]._indices()
                    e3_scores[idx][indic] = 0.0
                    if self.GPU:
                        e3_scores[idx][e3_idx] = torch.from_numpy(tmp).cuda()
                    else:
                        e3_scores[idx][e3_idx] = torch.from_numpy(tmp)
                    
                mrr, mr, hit = mrr_mr_hitk(e1_scores[idx], e1_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                
                mrr, mr, hit = mrr_mr_hitk(e2_scores[idx], e2_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                
                mrr, mr, hit = mrr_mr_hitk(e3_scores[idx], e3_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                
                count += 3
                
                
        if arch is None:
            logging.info('Test_MRR=%f, Test_MR=%f, Test_H=%f %f %f, Count=%d', float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count, count)
        
        return float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count#, embed_time, mrr_time
    
    
    
    
    
    
    """
    def evaluate(self, test_data):
        hits, ranks, losses = [], [], []
        for _ in [1, 3, 10]:
            hits.append([])
        
        ary = test_data.size(1) - 1
                        
        er_vocab_list = []
        er_vocab_pairs_list = []
        for miss_ent_domain in range(1, ary+1):
            er_vocab = self.get_er_vocab(test_data, miss_ent_domain)
            er_vocab_pairs = list(er_vocab.keys())
            er_vocab_list.append(er_vocab)
            er_vocab_pairs_list.append(er_vocab_pairs)
            
        max_idx = torch.Tensor(self.asng.p_model.theta).argmax(1)
        
        for miss_ent_domain in range(1, ary+1):
            er_vocab = er_vocab_list[miss_ent_domain-1]

            #for i in range(0, len(test_data_idxs), self.batch_size):
            for facts in batch_by_size(self.args.test_batch_size, test_data):
                
                
                if self.GPU:
                    r_idx, e1_idx, e2_idx, e3_idx = facts[:,0].cuda(), facts[:,1].cuda(), facts[:,2].cuda(), facts[:,3].cuda()
                else:
                    r_idx, e1_idx, e2_idx, e3_idx = facts[:,0], facts[:,1], facts[:,2], facts[:,3]
                    
                    
                length = self.n_dim // self.K
                r_embed = self.model.rel_embed(r_idx).view(-1, self.K, length)
                e1_embed = self.model.ent_embed(e1_idx).view(-1, self.K, length)
                e2_embed = self.model.ent_embed(e2_idx).view(-1, self.K, length)
                e3_embed = self.model.ent_embed(e3_idx).view(-1, self.K, length)
                
                e1_scores = torch.sigmoid(self.model.tri_neg_other(r_embed, e2_embed, e3_embed, max_idx, 1)).data
                e2_scores = torch.sigmoid(self.model.tri_neg_other(r_embed, e1_embed, e3_embed, max_idx, 2)).data
                e3_scores = torch.sigmoid(self.model.tri_neg_other(r_embed, e1_embed, e2_embed, max_idx, 3)).data
                
                
                if ary == 3:
                    if miss_ent_domain == 1:
                        e_idx = [e2_idx, e3_idx]
                        pred = e1_scores
                    elif miss_ent_domain == 2:
                        e_idx = [e1_idx, e3_idx]
                        pred = e2_scores
                    elif miss_ent_domain == 3:
                        e_idx = [e1_idx, e2_idx]
                        pred = e3_scores
                         
                #pred, _ = model.forward(r_idx, e_idx, miss_ent_domain, W)
                
                
                
                e_all_idx = []
                for k0 in range(1, ary+1):
                    e_all_idx.append(torch.LongTensor(facts[:, k0]))
                

                #print(er_vocab)
                
                for j in range(facts.shape[0]):
                    er_vocab_key = []
                    for k0 in range(ary+1):
                        er_vocab_key.append(facts[j][k0])
                    er_vocab_key.remove(facts[j][miss_ent_domain])
                    
                    filt = er_vocab[tuple(er_vocab_key)]
                    
                    if filt != []:
                        print(er_vocab)
                    
#                    print(er_vocab)
#                    print(tuple(er_vocab_key))
#                    print(filt)
                    
                    target_value = pred[j, e_all_idx[miss_ent_domain-1][j]].item()
                    pred[j, filt] = 0.0
                    pred[j, e_all_idx[miss_ent_domain-1][j]] = target_value
                    
                    
                sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(facts.shape[0]):
                    rank = np.where(sort_idxs[j] == e_all_idx[miss_ent_domain-1][j].item())[0][0]
                    ranks.append(rank+1)
                    for id, hits_level in enumerate([1, 3, 10]):
                        if rank+1 <= hits_level:
                            hits[id].append(1.0)
                        else:
                            hits[id].append(0.0)
         

        #logging.info('Test_MRR=%f, Test_MR=%f, Test_H=%f %f %f', np.mean(1./np.array(ranks)), 0, np.mean(hits[0]), np.mean(hits[1]), np.mean(hits[2]))
        #return np.mean(1./np.array(ranks)), 0, np.mean(hits[0]), np.mean(hits[1]), np.mean(hits[2])

    
    
    def get_er_vocab(self, data, miss_ent_domain):
        er_vocab = defaultdict(list)
        if len(data[0])-1 == 3:
            if miss_ent_domain == 1:
                for triple in data:
                    er_vocab[(triple[0], triple[2], triple[3])].append(triple[1])
            elif miss_ent_domain == 2:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[3])].append(triple[2])
            elif miss_ent_domain == 3:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[2])].append(triple[3])
        elif len(data[0])-1 == 4:
            if miss_ent_domain == 1:
                for triple in data:
                    er_vocab[(triple[0], triple[2], triple[3], triple[4])].append(triple[1])
            elif miss_ent_domain == 2:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[3], triple[4])].append(triple[2])
            elif miss_ent_domain == 3:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[2], triple[4])].append(triple[3])
            elif miss_ent_domain == 4:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[2], triple[3])].append(triple[4])
        return er_vocab
    """
                
                

    """
    # triplet classification
    def test_trip_class(self, valid_trip_pos, valid_trip_neg, test_trip_pos, test_trip_neg):
        rela_thres = {}
        valid_pos = {}
        valid_neg = {}
        interval = 0.01
        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *valid_trip_pos):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if r_idx in valid_pos:
                    valid_pos[r_idx].append(s)
                else:
                    valid_pos[r_idx] = [s]

        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *valid_trip_neg):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if r_idx in valid_neg:
                    valid_neg[r_idx].append(s)
                else:
                    valid_neg[r_idx] = [s]

        for r in valid_pos:
            if not (r in valid_neg):
                continue
            min_score = min(valid_pos[r] + valid_neg[r])
            max_score = max(valid_pos[r] + valid_neg[r])
            n_interval = int((max_score - min_score) / interval)
            best_Thresh = 0
            for i in range(n_interval):
                tmpThresh = min_score + i*interval
                correct = 0
                for s in valid_pos[r]:
                    if s >= tmpThresh:
                        correct += 1
                for s in valid_neg[r]:
                    if s < tmpThresh:
                        correct += 1
                tmpAcc = 1.0 * correct / (len(valid_pos[r]) + len(valid_neg[r]))
                if i==0:
                    bestThresh = tmpThresh
                    bestAcc = tmpAcc
                elif tmpAcc > bestAcc:
                    bestAcc = tmpAcc
                    bestThresh = tmpThresh
            rela_thres[r] = bestThresh

        correct = 0
        total = 0
        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *test_trip_pos):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if not (r_idx in rela_thres):
                    continue
                total += 1
                if s >=  rela_thres[r_idx]:
                    correct += 1

        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *test_trip_neg):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if not (r_idx in rela_thres):
                    continue
                total += 1
                if s < rela_thres[r_idx]:
                    correct += 1
        return 100*float(correct)/total
    """
                
