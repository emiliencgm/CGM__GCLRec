"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import world
import torch
import numpy as np
import torch_scatter
from dataloader import dataset
import networkx as nx

#=============================================================Overall Precalculate============================================================#
class precalculate():
    def __init__(self, config, dataset):
        if self.if_exist_precalcul('popularity'):
            self.P = self.load_precalcul('popularity')
        else:
            self.P = Pop(dataset)

        if self.if_exist_precalcul('centroid'):
            self.C = self.load_precalcul('centroid')
        else:
            self.C = Centroid(dataset, self.P)

        if self.if_exist_precalcul('commonNeighbor'):
            self.CN = self.load_precalcul('commonNeighbor')
        else:
            self.CN = CommonNeighbor(dataset)
    
    def if_exist_precalcul(self,name):
        return False

    def load_precalcul(self, name):
        pass

    
    @property
    def popularity(self):
        return self.P
    
    @property
    def centroid(self):
        return self.C

    @property
    def common_neighbor(self):
        return self.CN
        
#=============================================================Popularity============================================================#
class Pop():
    """
    precalculate popularity of users and items
    """
    def __init__(self, dataset:dataset):
        self.TrainPop_item = dataset.TrainPop_item #item's popularity (degree) in the training dataset
        self.TrainPop_user = dataset.TrainPop_user #user's popularity (degree) in the training dataset
        self.num_item = dataset.m_items
        self.num_user = dataset.n_users
        self.UInet = dataset.UserItemNet
        self.testDict = dataset.testDict

        #self.pop_statistic()
        self._ItemPopGroupDict, self._reverse_ItemPopGroupDict, self._testDict_PopGroup = self.build_pop_item()
        self._UserPopGroupDict, self._reverse_UserPopGroupDict = self.build_pop_user()
        #self.pop_label()
        self._pop_bias_Dict = self.pop_bias()

    @property
    def ItemPopGroupDict(self):
        '''
        {
            group0 : tensor([item0, ..., itemN])
            group9 : tensor([item0, ..., itemM])
        }
        '''
        return self._ItemPopGroupDict

    @property
    def reverse_ItemPopGroupDict(self):
        '''
        {
            item0 : groupN
            item9 : groupM
        }
        '''
        return self._reverse_ItemPopGroupDict

    @property
    def testDict_PopGroup(self):
        '''
        {
            group0 : {user0 : [item0, ..., itemN ],  user9 : [item0, ..., itemM ]}
            group9 : {user0 : [item0, ..., itemN'],  user9 : [item0, ..., itemM']}
        }
        '''
        return self._testDict_PopGroup

    @property
    def UserPopGroupDict(self):
        '''
        {
            group0 : tensor([user0, ..., userN])
            group9 : tensor([user0, ..., userM])
        }
        '''
        return self._UserPopGroupDict

    @property
    def reverse_UserPopGroupDict(self):
        '''
        {
            user0 : groupN
            user9 : groupM
        }
        '''
        return self._reverse_UserPopGroupDict
        
    @property
    def pop_bias_Dict(self):
        '''
        Not Implemented
        '''
        return self._pop_bias_Dict

    @property
    def item_pop_group_label(self):
        '''
        [pop_group_of_item_0, ..., pop_group_of_item_999]
        '''
        return self._item_pop_label
    
    @property
    def item_pop_degree_label(self):
        '''
        [pop_degree_of_item_0, ..., pop_degree_of_item_999]
        '''
        return self._item_pop
    
    @property
    def user_pop_group_label(self):
        '''
        [pop_group_of_user_0, ..., pop_group_of_user_999]
        '''
        return self._user_pop_label

    @property
    def user_pop_degree_label(self):
        '''
        [pop_degree_of_user_0, ..., pop_degree_of_user_999]
        '''
        return self._user_pop


    def build_pop_item(self):
        num_group = world.config['pop_group']
        item_per_group = int(self.num_item / num_group)
        TrainPopSorted = sorted(self.TrainPop_item.items(), key=lambda x: x[1])
        ItemPopGroupDict = {}#查询分组中有哪些item的字典
        testDict_PopGroup = {}#查询不同分组下用户在Test集中交互过的item的字典
        reverse_ItemPopGroupDict = {}#查询item属于哪个分组的字典
        self._item_pop_label = [0]*self.num_item
        self._item_pop = [0]*self.num_item
        #按照Pop分组，并存储至字典[0=Cold, 9=Hot]
        for group in range(num_group):
            ItemPopGroupDict[group] = []
            if group == num_group-1:
                for item, pop in TrainPopSorted[group * item_per_group:]:
                    ItemPopGroupDict[group].append(item)
                    reverse_ItemPopGroupDict[item] = group
                    self._item_pop_label[item] = group
                    self._item_pop[item] = pop
            else:
                for item, pop in TrainPopSorted[group * item_per_group: (group+1) * item_per_group]:
                    ItemPopGroupDict[group].append(item)
                    reverse_ItemPopGroupDict[item] = group
                    self._item_pop_label[item] = group
                    self._item_pop[item] = pop
        self._item_pop_label = np.array(self._item_pop_label)
        #转换为tensor格式
        for group, items in ItemPopGroupDict.items():
            ItemPopGroupDict[group] = torch.tensor(items)

        #初始化testDict_PopGroup的格式：testDict_PopGroup={0:{user:ColdItem}}
        for group in range(num_group):
            testDict_PopGroup[group] = {}
        #生成不同热度分组下的用户test交互item字典
        for user, items in self.testDict.items():                
            Hot = {}
            for group in range(num_group):
                Hot[group] = []
            for item in items:
                group = reverse_ItemPopGroupDict[item]
                Hot[group].append(item)
            for group in range(num_group):
                if Hot[group]:
                    testDict_PopGroup[group][user] = Hot[group]
                else:
                    testDict_PopGroup[group][user] = [999999999999999]#缺省值
        #print(testDict_PopGroup[0])
        return ItemPopGroupDict, reverse_ItemPopGroupDict, testDict_PopGroup

    def build_pop_user(self):
        num_group = world.config['pop_group']
        user_per_group = int(self.num_user / num_group)
        TrainPopSorted = sorted(self.TrainPop_user.items(), key=lambda x: x[1])
        UserPopGroupDict = {}#查询分组中有哪些item的字典
        reverse_UserPopGroupDict = {}#查询item属于哪个分组的字典
        self._user_pop_label = [0]*self.num_user
        self._user_pop = [0]*self.num_user
        #按照Pop分组，并存储至字典[0=Cold, 9=Hot]
        for group in range(num_group):
            UserPopGroupDict[group] = []
            if group == num_group-1:
                for user, pop in TrainPopSorted[group * user_per_group:]:
                    UserPopGroupDict[group].append(user)
                    reverse_UserPopGroupDict[user] = group
                    self._user_pop_label[user] = group
                    self._user_pop[user] = pop
            else:
                for user, pop in TrainPopSorted[group * user_per_group: (group+1) * user_per_group]:
                    UserPopGroupDict[group].append(user)
                    reverse_UserPopGroupDict[user] = group
                    self._user_pop_label[user] = group
                    self._user_pop[user] = pop
        self._user_pop_label = np.array(self._user_pop_label)
        #转换为tensor格式
        for group, users in UserPopGroupDict.items():
            UserPopGroupDict[group] = torch.tensor(users)
        
        return UserPopGroupDict, reverse_UserPopGroupDict

    def pop_bias(self):
        '''
        for M_xy in terms of popularity
        '''
        pop_bias_Dict = {}
        return pop_bias_Dict

#=============================================================Node & Edge Centroid============================================================#
class Centroid():
    def __init__(self, dataset:dataset, pop:Pop):
        self.dataset = dataset
        self.pop = pop
        self._degree_item = torch.tensor(self.pop.item_pop_degree_label) #item's popularity (degree) in the training dataset
        self._degree_user = torch.tensor(self.pop.user_pop_degree_label) #user's popularity (degree) in the training dataset
        self._pagerank_user, self._pagerank_item = self.compute_pr()
        self._eigenvector_user, self._eigenvector_item = self.eigenvector_centrality()
        # look = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])
        # print('degree_user', self._degree_user[look])
        # print('degree_item', self._degree_item[look])
        # print('PR_user',self._pagerank_user[look])
        # print('PR_item',self._pagerank_item[look])
        # print('eigenvector_user',self._eigenvector_user[look])
        # print('eigenvector_item',self._eigenvector_item[look])
    
    @property
    def degree_weight_item(self):
        '''
        {item : degree_weight}
        '''
        return 

    @property
    def degree_weight_user(self):
        '''
        {user : degree_weight}
        '''
        return 

    @property
    def eigenvector_weight_item(self):
        '''
        
        '''
        return 

    @property
    def eigenvector_weight_user(self):
        '''
        
        '''
        return 

    @property
    def pagerank_weight_item(self):
        '''
        tensor([pr_i0, ..., pr_in])
        '''
        return 

    @property
    def pagerank_weight_user(self):
        '''
        tensor([pr_u0, ..., pr_un])
        '''
        return 

    def compute_pr(self, damp=0.85, k=10):
        '''
        For undirected graph, calculate twice pagerank in two direction\n
        U:|0   U-I|
        I:|I-U   0|
        '''
        num_nodes = self.dataset.n_users + self.dataset.m_items
        deg_out = torch.cat((torch.tensor(self.pop.user_pop_degree_label), torch.tensor(self.pop.item_pop_degree_label)))
        PR = torch.ones((num_nodes, )).to(torch.float32)
        nodes_out = torch.cat((torch.tensor(self.dataset._trainUser), torch.tensor(self.dataset._trainItem+self.dataset.n_users)))
        nodes_in =  torch.cat((torch.tensor(self.dataset._trainItem+self.dataset.n_users), torch.tensor(self.dataset._trainUser)))

        for i in range(k):
            edge_msg = PR[nodes_out] / deg_out[nodes_out]
            agg_msg = torch_scatter.scatter(edge_msg, nodes_in, reduce='sum')

            PR = (1 - damp) * PR + damp * agg_msg

        users_pagerank, items_pagerank = torch.split(PR, [self.dataset.n_users, self.dataset.m_items])
        return users_pagerank, items_pagerank

    def eigenvector_centrality(self):
        nx_graph = self.dataset.nx_Graph #TODO 在dataset中用额外的nx_Graph存储networkx格式有点浪费内存！
        x = nx.eigenvector_centrality(nx_graph, max_iter=100, tol=5e-04)

        num_nodes = self.dataset.n_users + self.dataset.m_items
        x = torch.tensor([x[i] for i in range(num_nodes)])
        #x = torch.tensor(x)
        eigenvector_centrality_user, eigenvector_centrality_item = torch.split(x, [self.dataset.n_users, self.dataset.m_items])
        return eigenvector_centrality_user, eigenvector_centrality_item

    def cal_weight_GCA(self, mode):
        return 

#=============================================================Commmon Neighborhood============================================================#
class CommonNeighbor():
    def __init__(self, dataset):
        pass