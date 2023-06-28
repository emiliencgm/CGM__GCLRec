"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import torch
import numpy as np
import world
import utils
import multiprocessing
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import LightGCN
from augment import Projector
from collections import OrderedDict
import loss
import time

class Train():
    def __init__(self, loss_cal):
        self.loss = loss_cal
        self.projector = Projector().to(world.device)
        self.test = Test()
        self.INFONCE = loss.InfoNCE_loss()
        self.BPR = loss.BPR()

    def train(self, dataset, Recmodel, augmentation, epoch, optimizer):
        Recmodel:LightGCN = Recmodel
        batch_size = world.config['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)#每个batch为batch_size对(user, pos_item, neg_item), 见Dataset.__getitem__

        total_batch = len(dataloader)
        aver_loss = 0.

        for batch_i, train_data in tqdm(enumerate(dataloader), desc='training'):
            batch_users = train_data[0].long().to(world.device)
            batch_pos = train_data[1].long().to(world.device)
            batch_neg = train_data[2].long().to(world.device)
            
            l_all = self.train_all(Recmodel, augmentation, batch_users, batch_pos, batch_neg, optimizer, epoch)

            aver_loss += l_all.cpu().item()
        aver_loss = aver_loss / (total_batch)
        print(f'EPOCH[{epoch}]:loss {aver_loss:.3f}')
        return aver_loss
    
    def train_batch(self, Recmodel, augmentation, batch_users, batch_pos, batch_neg, optimizer, epoch):
        #========================train Augmentation==========================
        #计算增强视图下的表示
        Recmodel.eval()
        augmentation.train()
        for param in Recmodel.parameters():
                param.requires_grad = False
        for param in augmentation.parameters():
                param.requires_grad = True
        for param in self.loss.MLP_model.parameters():
                param.requires_grad = True


        users_emb0 = Recmodel.embedding_user.weight.clone()
        items_emb0 = Recmodel.embedding_item.weight.clone()
        x = torch.cat([users_emb0, items_emb0])
        edge_index = Recmodel.edge_index
        # edge_weight = augmentation.forward()#参数更新 TODO .detach()  batch_fashion
        batch_x = torch.cat([users_emb0[batch_users], items_emb0[batch_pos]])
        src = torch.cat([batch_users, batch_pos+Recmodel.num_users])
        dst = torch.cat([batch_pos+Recmodel.num_users, batch_users])
        src = list(np.array(src.cpu()))
        dst = list(np.array(dst.cpu()))
        batch_edge_index = torch.tensor([src,
                                        dst])
        batch_edge_weight = augmentation.forward_batch(batch_x, batch_edge_index)
        batch_aug_users, batch_aug_items = Recmodel.view_computer(batch_x, batch_edge_index, edge_weight=batch_edge_weight)
        #使用增强表示进行推荐，计算Adaloss来更新Learner_Aug
        #TODO Learner_Aug的参数正则化项
        loss_aug = self.loss.adaptive_softmax_loss(batch_aug_users, batch_aug_items, None, batch_users, batch_pos, None, None, None, None, None, epoch)
        optimizer['aug'].zero_grad()
        loss_aug.backward(retain_graph=True)
        optimizer['aug'].step()


        #========================train Embeddings==========================
        #使用BPR_Contrast来训练Embedding
        Recmodel.train()
        augmentation.eval()
        for param in Recmodel.parameters():
                param.requires_grad = True
        for param in augmentation.parameters():
                param.requires_grad = False
        for param in self.loss.MLP_model.parameters():
                param.requires_grad = False

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs = Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())
        reg = (0.5 * torch.norm(userEmb0) ** 2 + len(batch_pos) * 0.5 * torch.norm(posEmb0) ** 2)/len(batch_pos)
        loss_bpr = self.BPR.bpr_loss(users_emb, pos_emb, neg_emb)
        aug_users1, aug_items1 = embs_per_layer_or_all_embs[0], embs_per_layer_or_all_embs[1]
        batch_aug_users1, batch_aug_items1 = aug_users1[batch_users], aug_items1[batch_pos]
        users_emb0 = Recmodel.embedding_user.weight
        items_emb0 = Recmodel.embedding_item.weight
        # x = torch.cat([users_emb0, items_emb0])
        batch_x = torch.cat([users_emb0[batch_users], items_emb0[batch_pos]])
        batch_aug_users2, batch_aug_items2 = Recmodel.view_computer(batch_x, batch_edge_index, edge_weight=batch_edge_weight.detach())
        loss_infonce = self.INFONCE.infonce_loss_batch(batch_aug_users1, batch_aug_items1, batch_aug_users2, batch_aug_items2)            
        loss_emb = world.config['weight_decay']*reg + loss_bpr + world.config['lambda1']*loss_infonce
        loss_emb.requires_grad_(True)
        #固定Learner_Aug的参数再更新Embedding
        optimizer['emb'].zero_grad()
        loss_emb.backward()
        optimizer['emb'].step()

        return loss_aug, loss_emb
    
    def train_all(self, Recmodel, augmentation, batch_users, batch_pos, batch_neg, optimizer, epoch):
        #========================train Embeddings==========================
        #使用BPR_Contrast来训练Embedding
        Recmodel.train()
        self.loss.MLP_model.train()
        augmentation.eval()
        optimizer['emb'].zero_grad()
        # for param in Recmodel.parameters():
        #         param.requires_grad = True
        # for param in augmentation.parameters():
        #         param.requires_grad = False
        # for param in self.loss.MLP_model.parameters():
        #         param.requires_grad = False

        users_emb0 = Recmodel.embedding_user.weight
        items_emb0 = Recmodel.embedding_item.weight

        userEmb0,  posEmb0 = users_emb0[batch_users], items_emb0[batch_pos]
        reg = torch.mean(0.5 * torch.norm(userEmb0) ** 2 +  0.5 * torch.norm(posEmb0) ** 2)

        x = torch.cat([users_emb0, items_emb0])
        edge_index = Recmodel.edge_index
        users, items = Recmodel.view_computer(x, edge_index, edge_weight=None)

        users_emb, pos_emb, neg_emb = users[batch_users], items[batch_pos], items[batch_neg]
        # loss_bpr = self.BPR.bpr_loss(users_emb, pos_emb, neg_emb)
        loss_ada = self.loss.adaptive_softmax_loss(users_emb, pos_emb, None, batch_users, batch_pos, None, None, None, None, None, epoch)

        aug_users1, aug_items1 = users, items

        edge_weight = augmentation.forward(x.detach(), edge_index)#TODO detach()
        aug_users2, aug_items2 = Recmodel.view_computer(x, edge_index, edge_weight=edge_weight.detach())#TODO .detach()
        batch_aug_users, batch_aug_items = aug_users2[batch_users], aug_items2[batch_pos]

        # loss_infonce = self.INFONCE.infonce_loss(batch_users, batch_pos, aug_users1, aug_items1, aug_users2, aug_items2)  
        loss_ada_aug = self.loss.adaptive_softmax_loss(batch_aug_users, batch_aug_items, None, batch_users, batch_pos, None, None, None, None, None, epoch)          
        
        loss_emb = world.config['weight_decay']*reg + 0.1* loss_ada + 0.1* loss_ada_aug #+ world.config['lambda1']*loss_infonce
        loss_emb.requires_grad_(True)
        #固定Learner_Aug的参数再更新Embedding
        loss_emb.backward()
        optimizer['emb'].step()
        
        #========================train Augmentation==========================
        #计算增强视图下的表示
        Recmodel.eval()
        self.loss.MLP_model.eval()
        augmentation.train()
        optimizer['aug'].zero_grad()
        # for param in Recmodel.parameters():
        #         param.requires_grad = False
        # for param in augmentation.parameters():
        #         param.requires_grad = True
        # for param in self.loss.MLP_model.parameters():
        #         param.requires_grad = True


        # users_emb0 = Recmodel.embedding_user.weight.detach()
        # items_emb0 = Recmodel.embedding_item.weight.detach()
        # x = torch.cat([users_emb0, items_emb0])
        # edge_index = Recmodel.edge_index
        # edge_weight = augmentation.forward()#参数更新 TODO .detach()  batch_fashion
        # aug_users, aug_items = Recmodel.view_computer(x, edge_index, edge_weight=edge_weight)
        # batch_aug_users, batch_aug_items = aug_users[batch_users], aug_items[batch_pos]
        # #使用增强表示进行推荐，计算Adaloss来更新Learner_Aug
        # #TODO Learner_Aug的参数正则化项
        # loss_aug = self.loss.adaptive_softmax_loss(batch_aug_users, batch_aug_items, None, batch_users, batch_pos, None, None, None, None, None, epoch)
        
        # optimizer['aug'].zero_grad()
        # loss_aug.backward()
        # optimizer['aug'].step()

        users_emb0 = Recmodel.embedding_user.weight.detach()
        items_emb0 = Recmodel.embedding_item.weight.detach()

        x = torch.cat([users_emb0, items_emb0])
        edge_index = Recmodel.edge_index
        users, items = Recmodel.view_computer(x, edge_index, edge_weight=None)
        edge_weight = augmentation.forward(x, edge_index)#TODO detach()
        aug_users, aug_items = Recmodel.view_computer(x, edge_index, edge_weight=edge_weight)

        loss_infonce = -self.INFONCE.infonce_loss(batch_users, batch_pos, users, items, aug_users, aug_items)

        loss_instance = self.calc_instance_loss(users[batch_users], aug_users[batch_users]) + self.calc_instance_loss(items[batch_pos], aug_items[batch_pos])

        # # current parameter
        # fast_weights = OrderedDict((name, param) for (name, param) in Recmodel.named_parameters())

        # # create_graph flag for computing second-derivative
        # grads = torch.autograd.grad(loss_infonce, Recmodel.parameters(), create_graph=True)
        # data = [p.data for p in list(Recmodel.parameters())]

        # # compute parameter' by applying sgd on multi-task loss
        # fast_weights = OrderedDict( (name, param - world.config['lr'] * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data) )
        
        # # compute primary loss with the updated parameter'
        # with torch.no_grad():
        #     users_emb0 = fast_weights['embedding_user.weight']
        #     items_emb0 = fast_weights['embedding_item.weight']
        #     x = torch.cat([users_emb0, items_emb0])
        #     edge_index = Recmodel.edge_index
        #     users, items = Recmodel.view_computer(x, edge_index, edge_weight=None)
        # aug_users, aug_items = Recmodel.view_computer(x, edge_index, edge_weight=edge_weight.detach())
        
        # loss_aug = self.calc_instance_loss(users[batch_users], aug_users[batch_users]) + self.calc_instance_loss(items[batch_pos], aug_items[batch_pos])
        loss_aug = 0.* loss_infonce + 0.1* loss_instance
        loss_aug.requires_grad_(True)
        loss_aug.backward()
        optimizer['aug'].step()

        # print('grad', grads)
        # print('')
        # for (name, param) in augmentation.named_parameters():
        #      print(name, param.grad)
        # print('')
        # print('reg',reg)
        # print('loss_ada',loss_ada)
        # print('loss_ada_aug',loss_ada_aug)
        # print('loss_infonce', loss_infonce)
        # print('loss_instance',loss_instance)
        # print('loss_emb', loss_emb)
        # print('loss_aug', loss_aug)
        return loss_aug + loss_emb
    
    def blindly_train(self, Recmodel, augmentation, batch_users, batch_pos, batch_neg, optimizer, epoch):
        '''
        搁这瞎训
        '''
        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs = Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())
        reg = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/len(batch_pos)
        loss_bpr = self.BPR.bpr_loss(users_emb, pos_emb, neg_emb)

        aug_users2, aug_items2 = embs_per_layer_or_all_embs[0].detach(), embs_per_layer_or_all_embs[1].detach()

        edge_weight = augmentation.forward()#参数更新 TODO .detach()  batch_fashion
        users_emb0 = Recmodel.embedding_user.weight.detach()
        items_emb0 = Recmodel.embedding_item.weight.detach()
        x = torch.cat([users_emb0, items_emb0])
        edge_index = Recmodel.edge_index
        # print(min(edge_weight))
        aug_users, aug_items = Recmodel.view_computer(x, edge_index, edge_weight=edge_weight)
        loss_infonce = self.INFONCE.infonce_loss(batch_users, batch_pos, aug_users, aug_items, aug_users2, aug_items2)
        # 添加L2正则化项
        l2_regularization = torch.tensor(0.).to(world.device)  # 初始化正则化项为0
        for param in augmentation.parameters():
            l2_regularization += torch.norm(param, p=2)  # 计算参数的L2范数并累加            
        loss_emb = world.config['weight_decay']*reg + loss_bpr + world.config['lambda1']*loss_infonce + l2_regularization*world.config['weight_decay']
        optimizer['emb'].zero_grad()
        loss_emb.backward()
        optimizer['emb'].step()

        return loss_emb


    def train_BPR(self, Recmodel, augmentation, batch_users, batch_pos, batch_neg, optimizer, epoch):
        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs = Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())
        reg = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/len(batch_pos)
        loss_bpr = self.BPR.bpr_loss(users_emb, pos_emb, neg_emb)    
        loss_emb = world.config['weight_decay']*reg + loss_bpr 
        optimizer['emb'].zero_grad()
        loss_emb.backward()
        optimizer['emb'].step()
        return loss_emb

    def calc_instance_loss(self, x, x_aug):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = 2 * pos_sim.sum().div(batch_size * batch_size) - sim_matrix.sum().div(batch_size * batch_size)

        return loss

class Test():
    def __init__(self):
        pass
    
    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        #================Pop=================#
        groundTrue_popDict = X[2]#{0: [ [items of u1], [items of u2] ] }
        r, r_popDict = utils.getLabel(groundTrue, groundTrue_popDict, sorted_items)
        #================Pop=================#
        pre, recall, recall_pop, recall_pop_Contribute, ndcg = [], [], {}, {}, []
        num_group = world.config['pop_group']
        for group in range(num_group):
                recall_pop[group] = []
        for group in range(num_group):
                recall_pop_Contribute[group] = []

        for k in world.config['topks']:
            ret = utils.RecallPrecision_ATk(groundTrue, groundTrue_popDict, r, r_popDict, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])

            num_group = world.config['pop_group']
            for group in range(num_group):
                recall_pop[group].append(ret['recall_popDIct'][group])
            for group in range(num_group):
                recall_pop_Contribute[group].append(ret['recall_Contribute_popDict'][group])

            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))

        
        for group in range(num_group):
            recall_pop[group] = np.array(recall_pop[group])
        for group in range(num_group):
            recall_pop_Contribute[group] = np.array(recall_pop_Contribute[group])

        return {'recall':np.array(recall), 
                'recall_popDict':recall_pop,
                'recall_Contribute_popDict':recall_pop_Contribute,
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}


    def test(self, dataset, Recmodel, precal, epoch, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        testDict: dict = dataset.testDict
        testDict_pop = precal.popularity.testDict_PopGroup
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])
        CORES = multiprocessing.cpu_count() // 2
        # CORES = multiprocessing.cpu_count()
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.config['topks'])),
                'recall': np.zeros(len(world.config['topks'])),
                'recall_pop': {},
                'recall_pop_Contribute': {},
                'ndcg': np.zeros(len(world.config['topks']))}
        num_group = world.config['pop_group']
        for group in range(num_group):
            results['recall_pop'][group] = np.zeros(len(world.config['topks']))
            results['recall_pop_Contribute'][group] = np.zeros(len(world.config['topks']))

        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            groundTrue_list_pop = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                #================Pop=================#
                groundTrue_pop = {}
                for group, ground in testDict_pop.items():
                    groundTrue_pop[group] = [ground[u] for u in batch_users]
                #================Pop=================#
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [ 
                #         utils.AUC(rating[i],
                #                   dataset, 
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
                #================Pop=================#
                groundTrue_list_pop.append(groundTrue_pop)
                #================Pop=================#
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list, groundTrue_list_pop)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size/len(users))
                
            for result in pre_results:
                results['recall'] += result['recall']
                for group in range(num_group):
                    results['recall_pop'][group] += result['recall_popDict'][group]
                    results['recall_pop_Contribute'][group] += result['recall_Contribute_popDict'][group]
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            for group in range(num_group):
                results['recall_pop'][group] /= float(len(users))
                results['recall_pop_Contribute'][group] /= float(len(users))

            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if multicore == 1:
                pool.close()
            print(results)
            return results
    

    def valid_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r= utils.getLabel_Valid(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []

        for k in world.config['topks']:
            ret = utils.RecallPrecision_ATk_Valid(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall),
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}
    
    def valid_one_batch_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r= utils.getLabel_Valid(groundTrue, sorted_items)

        k = world.config['topks'][0]
        ret = utils.RecallPrecision_ATk_Valid(groundTrue, r, k)
        recall = ret['recall']
        return recall

    def valid(self, dataset, Recmodel, multicore=0, if_print=True):
        u_batch_size = world.config['test_u_batch_size']
        validDict: dict = dataset.validDict
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])
        CORES = multiprocessing.cpu_count() // 2
        # CORES = multiprocessing.cpu_count()
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.config['topks'])),
                'recall': np.zeros(len(world.config['topks'])),
                'ndcg': np.zeros(len(world.config['topks']))}

        with torch.no_grad():
            users = list(validDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                if if_print:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [validDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [ 
                #         utils.AUC(rating[i],
                #                   dataset, 
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.valid_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.valid_one_batch(x))
            scale = float(u_batch_size/len(users))
                
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if multicore == 1:
                pool.close()
            if if_print:
                print('VALID',results)
            return results


    def valid_batch(self, dataset, Recmodel, batch_users):
        batch_users = batch_users.cpu()
        validDict: dict = dataset.validDict
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])

        with torch.no_grad():
            users = list(batch_users)
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [validDict[u.item()] for u in batch_users]
            batch_users_gpu = batch_users.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            recall = self.valid_one_batch_batch([rating_K.cpu(), groundTrue])
            return recall/float(len(users))