"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
import numpy as np
from utils import randint_choice
import scipy.sparse as sp
import world
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from precalcul import precalculate
import time
from k_means import kmeans
import faiss


class Homophily:
    def __init__(self, model:LightGCN):
        self.model = model
        
    def get_homophily_batch(self, batch_user:torch.Tensor, batch_item:torch.Tensor, mode='not_in_batch'):
        '''
        return prob distribution of users and items in batch.
        '''
        with torch.no_grad():
            sigma = world.config['sigma_gausse']
            ncluster = world.config['n_cluster']
            #edge_index = self.model.dataset.Graph.cpu().indices()
            if mode == 'in_batch':
                embs_KMeans = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            else:
                embs_KMeans = torch.cat((self.model.embedding_user.weight, self.model.embedding_item.weight), dim=0)
            
            if ncluster > 99:
                embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
                kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
                kmeans_faiss.train(embs_KMeans_numpy)
                centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
            else:
                cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
                centroids = cluster_centers.to(world.device)            
            
            logits = []
            embs_batch = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            for c in centroids:
                logits.append((-torch.square(embs_batch - c).sum(1)/sigma).view(-1, 1))
            logits = torch.cat(logits, axis=1)
            probs = F.softmax(logits, dim=1)
            #probs = F.normalize(logits, dim=1)# TODO
            #loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
            batch_user_prob, batch_item_prob = torch.split(probs, [batch_user.shape[0], batch_item.shape[0]])
        return batch_user_prob, batch_item_prob

    def get_homophily_batch_any(self, batch_embs1:torch.Tensor, batch_embs2:torch.Tensor):
        '''
        return prob distribution of users and items in batch.
        '''
        sigma = world.config['sigma_gausse']
        ncluster = world.config['n_cluster']
        #edge_index = self.model.dataset.Graph.cpu().indices()
        embs_KMeans = torch.cat((batch_embs1, batch_embs2), dim=0)
        
        if ncluster > 99:
            embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
            kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
            kmeans_faiss.train(embs_KMeans_numpy)
            centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
        else:
            cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
            centroids = cluster_centers.to(world.device)            
        
        logits = []
        for c in centroids:
            logits.append((-torch.square(embs_KMeans - c).sum(1)/sigma).view(-1, 1))
        logits = torch.cat(logits, axis=1)
        probs = F.softmax(logits, dim=1)
        #probs = F.normalize(logits, dim=1)# TODO
        #loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
        batch_prob1, batch_prob2 = torch.split(probs, [batch_embs1.shape[0], batch_embs2.shape[0]])
        
        return batch_prob1, batch_prob2


class ED_Uniform():
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.augAdjMatrix1 = None
        self.augAdjMatrix2 = None

    def Edge_drop_random(self, p_drop):
        '''
        return: dropout后保留的交互构成的按度归一化的邻接矩阵(sparse)
        '''
        n_nodes = self.model.num_users + self.model.num_items
        #注意数组复制问题！
        trainUser = self.model.dataset.trainUser.copy()
        trainItem = self.model.dataset.trainItem.copy()
        keep_idx = randint_choice(len(self.model.dataset.trainUser), size=int(len(self.model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = trainUser[keep_idx]
        item_np = trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if self.config['if_big_matrix']:
            g = self.model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            g = self.model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Edge_drop_random(p_drop)
        self.augAdjMatrix2 =  self.Edge_drop_random(p_drop)



class RW_Uniform(ED_Uniform):
    def __init__(self, config, model, precal, homophily):
        super(RW_Uniform, self).__init__(config, model, precal, homophily)

    def Random_Walk(self, p_drop):
        aug_g = []
        for layer in self.config['num_layers']:
            aug_g.append(self.Edge_drop_random(p_drop))
        return aug_g

    # def computer(self, p_drop):
    #     aug_g = self.Random_Walk(p_drop)
    #     return self.model.view_computer(aug_g)

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Random_Walk(p_drop)
        self.augAdjMatrix2 =  self.Random_Walk(p_drop)




class Adaptive_Neighbor_Augment:
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.L = self.config['num_layers']
        self.epsilon = self.config['epsilon_GCLRec']
        self.w = self.config['w_GCLRec']
    
    def get_adaptive_neighbor_augment(self, embs_per_layer, batch_users, batch_pos, batch_neg, k):
        '''
        return aug_all_users, aug_all_items of selected k-th layer\n
        u'(k) = (1-𝜀)*u(k) + (𝜀(L-k)/L)*u(L) + w Σ w_uv*v(L)
        '''
        aug_embs_k_layer = (1-self.epsilon) * embs_per_layer[k] + (self.epsilon*(self.L-k)/self.L) * embs_per_layer[self.L]
        Sigma = 0
        
        aug_embs_k_layer = aug_embs_k_layer + self.w * Sigma

        aug_user_embs_k_layer, aug_item_embs_k_layer = torch.split(aug_embs_k_layer, [self.model.num_users, self.model.num_items])
        return aug_user_embs_k_layer, aug_item_embs_k_layer
        
    def get_adaptive_neighbor_augment_batch(self, embs_per_layer, batch_users, batch_pos, batch_neg, k):
        '''
        return aug_all_users, aug_all_items of selected k-th layer\n
        u'(k) = (1-𝜀)*u(k) + (𝜀(L-k)/L)*u(L) + wΣ w_uv*v(L)
        '''

        return 
        
    def sample(self):
        '''
        sample several samples for each user or item
        '''
        return 

