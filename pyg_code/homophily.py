import world
from model import LightGCN
import torch
import faiss
from kmeans_gpu import kmeans
import torch.nn.functional as F

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
            
            if ncluster > 64:
                embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
                kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
                kmeans_faiss.train(embs_KMeans_numpy)
                centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
            else:
                # cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
                cluster_ids_x, cluster_centers, dis = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device)
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
    def get_homophily_batch_epoch(self, batch_user:torch.Tensor, batch_item:torch.Tensor, epoch, mode='not_in_batch'):
        '''
        return prob distribution of users and items in batch.
        '''
        if epoch > 4:#TODO @5
            return self.batch_user_prob, self.batch_item_prob
        with torch.no_grad():
            sigma = world.config['sigma_gausse']
            ncluster = world.config['n_cluster']
            #edge_index = self.model.dataset.Graph.cpu().indices()
            if mode == 'in_batch':
                embs_KMeans = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            else:
                embs_KMeans = torch.cat((self.model.embedding_user.weight, self.model.embedding_item.weight), dim=0)
            
            if ncluster > 64:
                embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
                kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
                kmeans_faiss.train(embs_KMeans_numpy)
                centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
            else:
                # cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
                cluster_ids_x, cluster_centers, dis = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device)
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
            self.batch_user_prob, self.batch_item_prob = batch_user_prob, batch_item_prob
        return batch_user_prob, batch_item_prob

    def get_homophily_batch_any(self, embs_KMeans):
        '''
        return prob distribution of users and items in batch.
        '''
        sigma = world.config['sigma_gausse']
        ncluster = world.config['n_cluster']
        #edge_index = self.model.dataset.Graph.cpu().indices()
        # embs_KMeans = torch.cat((batch_embs1, batch_embs2), dim=0)
        
        if ncluster > 99:
            embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
            kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
            kmeans_faiss.train(embs_KMeans_numpy)
            centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
        else:
            # cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
            cluster_ids_x, cluster_centers, dis = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='cosine', device=world.device)
            centroids = cluster_centers.to(world.device)            
        
        logits = []
        for c in centroids:
            logits.append((-torch.square(embs_KMeans - c).sum(1)/sigma).view(-1, 1))
        logits = torch.cat(logits, axis=1)
        # probs = F.softmax(logits, dim=1)
        probs = F.normalize(logits, dim=1)# TODO
        #loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
        # batch_prob1, batch_prob2 = torch.split(probs, [batch_embs1.shape[0], batch_embs2.shape[0]])
        
        # return batch_prob1, batch_prob2
        return probs