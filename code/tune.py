#tune code currently
import os
cuda = 0
#for yelp2018

# os.system(f'python main.py --model LightGCN --loss Adaptive --augment No\
#             --init_method Normal --adaptive_method mlp --centroid_mode eigenvector --commonNeighbor_mode SC\
#             --temp_tau 0.1 --alpha 0.47\
#             --n_cluster 100 --sigma_gausse 1. --epsilon_GCLRec 0.1 --w_GCLRec 0.1 --k_aug 0\
#             --lr 0.001 --weight_decay 1e-4 --lambda1 0.1\
#             --if_visual 0 --cuda {cuda} --comment tune_augment\
#             --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --svd_q 5\
#             --early_stop_steps 40')

#model available: LightGCN, GTN, SGL, SimGCL, BCloss, GCLRec
#tuning loss only: change temp_tau, alpha, lr, weight_decay, lambda1, &&& adaptive coef implementation
#stop visualization currently if_visual 0
#try tuning alpha
#try tuning Augmentation
#max=0.0702: , Augment=No, LightGCN, loss=Adaptive(CV_mlp_projector, No_homophily, With_embs?, tau=0.1, alpha=0.47, No_Sigmoid?), Init_Normal



#for DCL loss   tau_plus to tune
# os.system(f'python main.py --model LightGCN --loss BC --augment No\
#             --init_method Normal --adaptive_method mlp --centroid_mode eigenvector --commonNeighbor_mode SC\
#             --temp_tau 0.1 --alpha 0.47\
#             --n_cluster 100 --sigma_gausse 1. --epsilon_GCLRec 0.1 --w_GCLRec 0.1 --k_aug 0\
#             --lr 0.001 --weight_decay 1e-4 --lambda1 0.1\
#             --if_visual 0 --cuda {cuda} --comment tune_DCL_loss\
#             --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --svd_q 5\
#             --early_stop_steps 40\
#             --tau_plus 0.001')



#for Softmax_loss in other datasets
os.system(f'python main.py --model LightGCN --loss BPR --augment No --dataset amazon-book\
            --init_method Normal --adaptive_method None --centroid_mode eigenvector --commonNeighbor_mode SC\
            --temp_tau 0.1 --alpha 0.5\
            --n_cluster 100 --sigma_gausse 1. --epsilon_GCLRec 0.1 --w_GCLRec 0.1 --k_aug 0\
            --lr 0.001 --weight_decay 1e-4 --lambda1 0.1\
            --if_visual 0 --cuda {cuda} --comment tune_BPR_amazon\
            --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --svd_q 5\
            --early_stop_steps 40\
            --tau_plus 0.001')
#tune_Softmax_b-1_using_BC_code