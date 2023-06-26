"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
from precalcul import precalculate
import torch
import torch.nn.functional as F
import world
from augment import Homophily
import numpy as np
import math
#=============================================================BPR loss============================================================#
class BPR():
    def __init__(self):
        return 
    def bpr_loss(self, users_emb, pos_emb, neg_emb):
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO SOFTPLUS()!!!
        return loss/world.config['batch_size']
#=============================================================BPR + CL loss============================================================#
class InfoNCE_loss():
    def __init__(self):
        self.tau = world.config['temp_tau']

    def infonce_loss(self, batch_user, batch_pos, aug_users1, aug_items1, aug_users2, aug_items2):

        # reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/self.config['batch_size']
        
        contrastloss = self.info_nce_loss_overall(aug_users1[batch_user], aug_users2[batch_user], aug_users2) \
                        + self.info_nce_loss_overall(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2)
        return contrastloss

    def info_nce_loss_overall(self, z1, z2, z_all):
        '''
        z1--z2: pos,  z_all: neg\n
        return: InfoNCEloss
        '''
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        all_sim = f(self.sim(z1, z_all))
        positive_pairs = (between_sim)
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))#TODO softplus
        #print('positive_pairs / negative_pairs',max(positive_pairs / negative_pairs))
        loss = loss/world.config['batch_size']
        return loss


    def sim(self, z1: torch.Tensor, z2: torch.Tensor, mode='inner_product'):#TODO
        '''
        计算一个z1和一个z2两个向量的相似度/或者一个z1和多个z2的各自相似度。
        即两个输入的向量数（行数）可能不同。
        '''
        if mode == 'inner_product':
            if z1.size()[0] == z2.size()[0]:
                #return F.cosine_similarity(z1,z2)
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                return torch.sum(torch.mul(z1,z2) ,dim=1)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())
        elif mode == 'cos':
            if z1.size()[0] == z2.size()[0]:
                return F.cosine_similarity(z1,z2)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())

#=============================================================Myself Adaptive loss============================================================#
class MLP(torch.nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()

        #1-->2-->2-->1
        # self.linear1=torch.nn.Linear(in_dim,3*in_dim)
        # self.activation1=torch.nn.ReLU()
        # self.linear2=torch.nn.Linear(3*in_dim, in_dim)
        # self.activation2=torch.nn.ReLU()
        # self.linear3=torch.nn.Linear(in_dim,1)
        #TODO from CV projector:
        self.linear = torch.nn.Linear(in_dim,4*in_dim)
        self.BatchNorm = torch.nn.BatchNorm1d(4*in_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.ELU()
        self.linear_hidden = torch.nn.Linear(4*in_dim,4*in_dim)
        self.linear_out = torch.nn.Linear(4*in_dim, 1)

    def forward(self, x):
        #TODO Architecture suboptimal
        # x = self.linear1(x)
        # x = self.activation1(x)
        # x = self.linear2(x)
        # x = self.activation2(x)
        # x = self.linear3(x)
        x = self.linear(x)
        # x = self.BatchNorm(x)
        x = self.activation(x)
        x = self.linear_hidden(x)
        x = self.activation(x)
        x = self.linear_out(x)
        x = torch.sigmoid(x)
        return x


class Adaptive_softmax_loss(torch.nn.Module):
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        super(Adaptive_softmax_loss, self).__init__()

        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.tau = config['temp_tau']
        self.alpha = config['alpha']
        self.f = lambda x: torch.exp(x / self.tau)
        self.MLP_model = MLP(5+2*0).to(world.device)
        self.MLP_model_CL = MLP(2+2*0).to(world.device)
        self.MLP_model_negative = MLP(3+2*0).to(world.device)

    def adaptive_softmax_loss(self, users_emb, pos_emb, neg_emb, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2, epoch):
        '''
        不再计算Regloss!
        '''
        loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb, batch_user, batch_pos, self.config['adaptive_method'], self.config['centroid_mode'], epoch)
        #loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb, None, None, None, None)
        if not (aug_users1 is None):
            loss2 = self.calculate_loss(aug_users1[batch_user], aug_users2[batch_user], aug_users2, None, None, None, None, epoch)
            loss3 = self.calculate_loss(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2, None, None, None, None, epoch)
            loss = loss1 + self.config['lambda1']*(loss2 + loss3)
        else:
            loss = loss1
        # loss_homophily = self.calculate_loss_homophily(users_emb, pos_emb)
        # loss = loss + loss_homophily
        
        return loss


    def calculate_loss(self, batch_target_emb, batch_pos_emb, batch_negs_emb, batch_target, batch_pos, method, mode, epoch):
        '''
        input : embeddings, not index.
        '''

        users_emb = batch_target_emb
        pos_emb = batch_pos_emb

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        # ratings_margin = self.get_coef_adaptive_all(batch_target_emb.detach(), batch_pos_emb.detach())
        # ratings = torch.cos(torch.arccos(torch.clamp(ratings,-1+1e-7,1-1e-7))+ratings_margin)
        ratings_diag = torch.diag(ratings)
        #UI
        if not (method is None):
            #Adaptive coef between User and Item
            if world.config['if_adaptive']:
                pos_ratings_margin = self.get_coef_adaptive(batch_target, batch_pos, method=method, mode=mode)
                theta = torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))
                M = torch.arccos(torch.clamp(pos_ratings_margin,-1+1e-7,1-1e-7))
                # M = torch.ones_like(M) - M
                # M = torch.clamp(M, torch.zeros_like(M), math.pi-theta)
                ratings_diag = torch.cos(theta + M)#TODO + or -
                # ratings_diag = ratings_diag * pos_ratings_margin
                #reliable / important ==> big margin ==> small theta ==> big simi between u,i 
            else:
                pass
                
            if world.config['adaloss_mode'] in ['pos_neg', 'pos_neg_cl']:
                neg_ratings_margin = self.get_coef_adaptive_negative(batch_target, batch_pos, method=method, mode=mode, epoch=epoch).squeeze()
                theta = torch.arccos(torch.clamp(ratings,-1+1e-7,1-1e-7))
                M = torch.arccos(torch.clamp(neg_ratings_margin,-1+1e-7,1-1e-7))
                ratings = torch.cos(theta+M)
        #UU, II
        else:
            if world.config['adaloss_mode'] in ['pos_neg_cl']:
                pos_ratings_margin = self.get_coef_adaptive_CL(batch_target, batch_target, epoch=epoch)
                theta = torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))
                M = torch.arccos(torch.clamp(pos_ratings_margin,-1+1e-7,1-1e-7))
                ratings_diag = torch.cos(theta + M)
            pass
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        #loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        loss = torch.mean(torch.negative((2*self.alpha * torch.log(numerator) -  2*(1-self.alpha) * torch.log(denominator))))
        #TODO trying torch.nn.functional.softplus 
        
        return loss

    def get_popdegree(self, batch_user, batch_pos_item):
        with torch.no_grad():
            pop_user = torch.tensor(self.precal.popularity.user_pop_degree_label).to(world.device)[batch_user]
            pop_item = torch.tensor(self.precal.popularity.item_pop_degree_label).to(world.device)[batch_pos_item]
        return pop_user, pop_item

    def get_homophily(self, batch_user, batch_pos_item):
        with torch.no_grad():
            batch_user_prob, batch_item_prob = self.homophily.get_homophily_batch(batch_user, batch_pos_item)
            batch_weight = torch.sum(torch.mul(batch_user_prob, batch_item_prob) ,dim=1)
        return batch_weight
    
    def get_homophily_CL(self, batch_user, batch_pos_item, epoch):
        with torch.no_grad():
            batch_user_prob, _ = self.homophily.get_homophily_batch_epoch(batch_user, batch_pos_item, epoch=epoch)
            batch_weight = torch.sum(torch.mul(batch_user_prob, batch_user_prob) ,dim=1)
        return batch_weight
    
    def get_homophily_negative(self, batch_user, batch_pos_item, epoch):
        '''
        shape: batch_u * batch_i
        '''
        with torch.no_grad():
            batch_user_prob, batch_item_prob = self.homophily.get_homophily_batch_epoch(batch_user, batch_pos_item, epoch=epoch)
            batch_weight = torch.matmul(batch_user_prob, torch.transpose(batch_item_prob, 0, 1))
        return batch_weight
    
    def get_centroid(self, batch_user, batch_pos_item, centroid='eigenvector', aggr='mean', mode='GCA'):
        with torch.no_grad():
            batch_weight = self.precal.centroid.cal_centroid_weights_batch(batch_user, batch_pos_item, centroid=centroid, aggr=aggr, mode=mode)
        return batch_weight
    
    def get_commonNeighbor(self, batch_user, batch_pos_item):
        with torch.no_grad():
            n_users = self.model.num_users
            csr_matrix_CN_simi = self.precal.common_neighbor.CN_simi_mat_sp
            batch_user, batch_pos_item = np.array(batch_user.cpu()), np.array(batch_pos_item.cpu())
            batch_weight1 = csr_matrix_CN_simi[batch_user, batch_pos_item+n_users]
            batch_weight2 = csr_matrix_CN_simi[batch_pos_item+n_users, batch_user]
            batch_weight1 = torch.tensor(np.array(batch_weight1).reshape((-1,))).to(world.device)
            batch_weight2 = torch.tensor(np.array(batch_weight2).reshape((-1,))).to(world.device)
        return batch_weight1, batch_weight2

    def get_embs_perturb(self, batch_user, batch_pos_item):
        with torch.no_grad():
            batch_weight_user, batch_weight_item = self.model.embedding_user(batch_user), self.model.embedding_item(batch_pos_item)
            low = torch.zeros_like(batch_weight_user).float()
            high = torch.ones_like(batch_weight_user).float()
            random_noise_user = torch.distributions.uniform.Uniform(low, high).sample()
            noise_user = torch.mul(torch.sign(batch_weight_user),torch.nn.functional.normalize(random_noise_user, dim=1)) * world.config['eps_SimGCL']
            batch_weight_user += noise_user
            random_noise_item = torch.distributions.uniform.Uniform(low, high).sample()
            noise_item = torch.mul(torch.sign(batch_weight_item),torch.nn.functional.normalize(random_noise_item, dim=1)) * world.config['eps_SimGCL']
            batch_weight_item += noise_item
        return batch_weight_user, batch_weight_item

    def get_mlp_input(self, features):
        '''
        features = [tensor, tensor, ...]
        '''
        U = features[0].unsqueeze(0)
        for i in range(1,len(features)):
            U = torch.cat((U, features[i].unsqueeze(0)), dim=0)
        return U.T
    
    def get_mlp_input_negative(self, features):
        '''
        features = [tensor, tensor, ...]
        '''
        U = features[0].unsqueeze(-1)
        for i in range(1,len(features)):
            U = torch.cat((U, features[i].unsqueeze(-1)), dim=2)
        return U


    def get_coef_adaptive(self, batch_user, batch_pos_item, method='mlp', mode='eigenvector'):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_n])\n
        the bigger, the more reliable, the more important
        '''
        if method == 'homophily':
            batch_weight = self.get_homophily(batch_user, batch_pos_item)
            batch_weight = 1. * batch_weight 

        elif method == 'centroid':
            batch_weight = self.get_centroid(batch_user, batch_pos_item, centroid=mode, aggr='mean', mode='GCA')
            batch_weight = 1. * batch_weight

        elif method == 'commonNeighbor':
            batch_weight1, batch_weight2 = self.get_commonNeighbor(batch_user, batch_pos_item)
            batch_weight = (batch_weight1 + batch_weight2)*0.5
            batch_weight = 1. * batch_weight

        elif method == 'mlp':
            # batch_weight_emb_user, batch_weight_emb_item = self.get_embs_perturb(batch_user, batch_pos_item)
            batch_weight_pop_user, batch_weight_pop_item = self.get_popdegree(batch_user, batch_pos_item)
            # batch_weight_pop_user = torch.ones_like(batch_weight_pop_user)*math.log(self.precal.popularity.max_pop_u)-torch.log(batch_weight_pop_user)#TODO problem of grandeur and +-
            # batch_weight_pop_item = torch.ones_like(batch_weight_pop_item)*math.log(self.precal.popularity.max_pop_i)-torch.log(batch_weight_pop_item)
            #batch_weight_homophily = self.get_homophily(batch_user, batch_pos_item)
            batch_weight_pop_user, batch_weight_pop_item = torch.log(batch_weight_pop_user), torch.log(batch_weight_pop_item)
            batch_weight_centroid = self.get_centroid(batch_user, batch_pos_item, centroid=mode, aggr='mean', mode='GCA')
            batch_weight_commonNeighbor1, batch_weight_commonNeighbor2 = self.get_commonNeighbor(batch_user, batch_pos_item)
            features = [batch_weight_pop_user, batch_weight_pop_item, batch_weight_centroid, batch_weight_commonNeighbor1, batch_weight_commonNeighbor2]
            
            # for i in range(self.config['latent_dim_rec']):
            #     features.append(batch_weight_emb_user[:,i])
            # for i in range(self.config['latent_dim_rec']):
            #     features.append(batch_weight_emb_item[:,i])
            
            batch_weight = self.get_mlp_input(features)
            batch_weight = self.MLP_model(batch_weight)

        else:
            batch_weight = None
            raise TypeError('adaptive method not implemented')
        
        self.batch_weight = batch_weight
        return batch_weight
    
    def get_coef_adaptive_CL(self, batch_user, batch_pos_user, method='mlp', mode='eigenvector', epoch=None):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_n])\n
        the bigger, the more reliable, the more important
        '''
        if method == 'mlp':
            batch_weight_pop_user, _ = self.get_popdegree(batch_user, batch_pos_user)
            batch_weight_pop_user= torch.log(batch_weight_pop_user)
            batch_weight_homophily = self.get_homophily_CL(batch_user, torch.tensor([0]), epoch=epoch)
            features = [batch_weight_pop_user, batch_weight_homophily]
            batch_weight = self.get_mlp_input(features)
            batch_weight = self.MLP_model_CL(batch_weight)

        else:
            batch_weight = None
            raise TypeError('adaptive method not implemented')
        
        return batch_weight


    def get_coef_adaptive_negative(self, batch_user, batch_pos_item, method='mlp', mode='eigenvector', epoch=None):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_m])\n
        ## shape: batch_u * batch_i\n
        the bigger, the more reliable, the more important
        '''
        if method == 'mlp':
            batch_weight_emb_user, batch_weight_emb_item = self.get_embs(batch_user, batch_pos_item)
            batch_weight_emb = torch.matmul(batch_weight_emb_user, torch.transpose(batch_weight_emb_item, 0, 1))
            batch_weight_pop_user, batch_weight_pop_item = self.get_popdegree(batch_user, batch_pos_item)
            batch_weight_pop_user, batch_weight_pop_item = torch.log(batch_weight_pop_user).unsqueeze(1), torch.log(batch_weight_pop_item).unsqueeze(1)
            all_ones = torch.ones_like(batch_weight_pop_user)
            batch_weight_pop_user = torch.matmul(batch_weight_pop_user, torch.transpose(all_ones, 0, 1))
            batch_weight_pop_item = torch.matmul(all_ones, torch.transpose(batch_weight_pop_item, 0, 1))
            batch_weight_homophily = self.get_homophily_negative(batch_user, batch_pos_item, epoch)
            features = [batch_weight_pop_user, batch_weight_pop_item, batch_weight_homophily]
            
            batch_weight = self.get_mlp_input_negative(features)
            batch_weight = self.MLP_model_negative(batch_weight)

        else:
            batch_weight = None
            raise TypeError('adaptive method not implemented')

        
        return batch_weight