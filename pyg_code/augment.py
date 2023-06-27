"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
from model import LGN_Encoder, GCN_Encoder
import scipy.sparse as sp
import world
import torch
from precalcul import precalculate
from dataloader import dataset
import copy
from homophily import Homophily


# ==================== 可学习的图数据增强：学得边的权重 ====================
class Augment_Learner(torch.nn.Module):
    def __init__(self, config, Recmodel:LightGCN, precal:precalculate, homophily:Homophily, dataset:dataset):
        super(Augment_Learner, self).__init__()
        self.config = config
        self.Recmodel = Recmodel        
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.trainUser = dataset._trainUser
        self.trainItem = dataset._trainItem
        self.num_edges = len(self.trainUser)
        self.src = torch.cat([torch.tensor(self.trainUser), torch.tensor(self.trainItem+self.num_users)])
        self.dst = torch.cat([torch.tensor(self.trainItem+self.num_users), torch.tensor(self.trainUser)])


        self.input_dim = self.config['latent_dim_rec']
        mlp_edge_model_dim = self.config['latent_dim_rec']

        self.GNN_encoder = GCN_Encoder(config['num_layers'], self.num_users, self.num_items)#TODO GCN or LGN encoder
        self.mlp_edge_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim * 2, mlp_edge_model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_edge_model_dim, 1),
            torch.nn.Sigmoid()
        )
        self.init_emb()
        
    def init_emb(self):#TODO 是否可以初始化GCN的权重？
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if world.config['init_method'] == 'Normal':
                    torch.nn.init.normal_(m.weight.data)
                elif world.config['init_method'] == 'Xavier':
                    torch.nn.init.xavier_uniform_(m.weight.data)
                else:
                    torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self):
        ''''
        返回增强后的边权重
        '''           
        users_emb0 = self.Recmodel.embedding_user.weight.detach()
        items_emb0 = self.Recmodel.embedding_item.weight.detach()
        x = torch.cat([users_emb0, items_emb0])
        edge_index =self.Recmodel.edge_index
        users_emb, items_emb = self.GNN_encoder.forward(x, edge_index)
        nodes_emb = torch.cat([users_emb, items_emb])

        emb_src = nodes_emb[self.src]
        emb_dst = nodes_emb[self.dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits #TODO .detach()
    
    def forward_batch(self, batch_x, batch_edge_index):
        users_emb, items_emb = self.GNN_encoder.forward(batch_x, batch_edge_index)
        node_emb = torch.cat([users_emb, items_emb])

        src, dst = batch_edge_index[0], batch_edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits
    



#================================CV projector==================================    
class Projector(torch.nn.Module):
    '''
    d --> Linear --> 2d --> BatchNorm --> ReLU --> 2d --> Linear --> d
    '''
    def __init__(self):
        super(Projector, self).__init__()

        self.input_dim = world.config['latent_dim_rec']
        self.linear1 = torch.nn.Linear(self.input_dim, 2*self.input_dim)
        self.BN = torch.nn.BatchNorm1d(2*self.input_dim)
        self.activate = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2*self.input_dim, self.input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.BN(x)
        x = self.activate(x)
        x = self.linear2(x)

        return x
    
#=====================================基于邻居进行增强=========================================
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
        with torch.no_grad():
            pass
            # TODO
        aug_embs_k_layer = (1-self.epsilon) * embs_per_layer[k] + (self.epsilon*(self.L-k)/self.L) * embs_per_layer[self.L]
        Sigma = 0

        # low = torch.zeros_like(aug_embs_k_layer).float()
        # high = torch.ones_like(aug_embs_k_layer).float()
        # random_noise = torch.distributions.uniform.Uniform(low, high).sample()
        # noise = torch.mul(torch.sign(aug_embs_k_layer),torch.nn.functional.normalize(random_noise, dim=1))
        
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
    