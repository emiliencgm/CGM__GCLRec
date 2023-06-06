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
# os.system(f'python main.py --model LightGCN --loss BPR --augment No --dataset amazon-book\
#             --init_method Normal --adaptive_method None --centroid_mode eigenvector --commonNeighbor_mode SC\
#             --temp_tau 0.1 --alpha 0.5\
#             --n_cluster 100 --sigma_gausse 1. --epsilon_GCLRec 0.1 --w_GCLRec 0.1 --k_aug 0\
#             --lr 0.001 --weight_decay 1e-4 --lambda1 0.1\
#             --if_visual 0 --cuda {cuda} --comment tune_BPR_amazon\
#             --num_layers 2 --latent_dim_rec 64 --batch_size 1024 --svd_q 5\
#             --early_stop_steps 40\
#             --tau_plus 0.001')
#tune_Softmax_b-1_using_BC_code
#change:batch_size==1024, num_layers==2


# os.system(f'python main.py --model LightGCN --loss Causal_pop --augment No --dataset gowalla\
#             --init_method Normal --adaptive_method None --centroid_mode eigenvector --commonNeighbor_mode SC\
#             --temp_tau 0.1 --alpha 0.5\
#             --n_cluster 100 --sigma_gausse 1. --epsilon_GCLRec 0.1 --w_GCLRec 0.1 --k_aug 0\
#             --lr 0.001 --weight_decay 1e-4 --lambda1 0.1\
#             --if_visual 0 --cuda {cuda} --comment tune_Causal_pop_BPR\
#             --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --svd_q 5\
#             --early_stop_steps 40\
#             --tau_plus 0.001\
#             --pop_gamma 0.02')


#Projector, temp_tau sensitive
# cuda = 1
# os.system(f'python main.py --model SimGCL --loss BPR_Contrast --augment No --dataset yelp2018\
#             --init_method Normal --adaptive_method None\
#             --temp_tau 0.2\
#             --if_visual 0 --cuda {cuda} --comment tune_Projector_SimGCL\
#             --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
#             --early_stop_steps 40\
#             --if_projector 0\
#             --if_valid 1')


#Aument_Learner and LightGCN_PyG
# os.system(f'python main.py --model LightGCN_PyG --loss BPR_Contrast --augment Learner --dataset yelp2018\
#             --init_method Normal --adaptive_method None\
#             --temp_tau 0.2\
#             --if_visual 0 --cuda {cuda} --comment tune_Aument_Learner_LGN_Contrast\
#             --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
#             --early_stop_steps 40\
#             --if_projector 0\
#             --if_valid 0')


# LightGCN: https://github.com/gusye1234/LightGCN-PyTorch/tree/master
# SGL: https://github.com/wujcan/SGL-TensorFlow/tree/main
# SimGCL: https://github.com/Coder-Yu/QRec
# GTN: https://github.com/wenqifan03/GTN-SIGIR2022
# BCloss: https://github.com/anzhang314/BC-Loss
# DCL: https://github.com/chingyaoc/DCL
# PDA: https://github.com/zyang1580/PDA
# #========================================YELP2018========================================
#Baseline--yelp2018--1 LightGCN + BPR
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+BPR --model LightGCN --loss BPR --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0 --notes _ --tag LightGCN --group baseline --job_type yelp2018')
# Baseline--yelp2018--2 LightGCN_PyG + BPR
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN_PyG+BPR --model LightGCN_PyG --loss BPR --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0 --notes PyG_Implementation --tag LightGCN --group baseline --job_type yelp2018')
#Baseline--yelp2018--3 SGL_ED+BPR_Contrast
os.system(f'python main.py --project GCLRec_No_Valid --name SGL_ED+BPR_CL --model SGL --loss BPR_Contrast --augment ED --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0 --notes _ --tag SGL --group baseline --job_type yelp2018')
#Baseline--yelp2018--4 SGL_RW+BPR_Contrast
os.system(f'python main.py --project GCLRec_No_Valid --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0 --notes _ --tag SGL --group baseline --job_type yelp2018')
#Baseline--yelp2018--5 SimGCL+BPR_Contrast
os.system(f'python main.py --project GCLRec_No_Valid --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 2 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.5 --temp_tau 0.2 --eps_SimGCL 0.1\
            --comment _ --if_valid 0 --notes layers=2 --tag SimGCL --group baseline --job_type yelp2018')#Layer=2
#Baseline--yelp2018--6 SimGCL+BPR_Contrast
os.system(f'python main.py --project GCLRec_No_Valid --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.5 --temp_tau 0.2 --eps_SimGCL 0.1\
            --comment _ --if_valid 0 --notes _ --tag SimGCL --group baseline --job_type yelp2018')#Layer=3
#Baseline--yelp2018--7 GTN + BPR
os.system(f'python main.py --project GCLRec_No_Valid --name GTN+BPR --model GTN --loss BPR --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0 --notes _ --tag GTN --group baseline --job_type yelp2018')
#Baseline--yelp2018--8 LightGCN + BC_loss
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset yelp2018\
            --init_method Normal --lr 5e-4 --weight_decay 1e-5 --cuda {cuda} --num_layers 2 --latent_dim_rec 64 --batch_size 2048\
            --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.07\
            --comment _ --if_valid 0 --notes layer=2 --tag BC_loss --group baseline --job_type yelp2018')#Layer=2, alpha for combing pop_loss and bc_loss
#Baseline--yelp2018--9 LightGCN + BC_loss
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset yelp2018\
            --init_method Normal --lr 5e-4 --weight_decay 1e-5 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.07\
            --comment _ --if_valid 0 --notes _ --tag BC_loss --group baseline --job_type yelp2018')#Layer=3, alpha for combing pop_loss and bc_loss
#Baseline--yelp2018--10 LightGCN + DCL
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+DCL --model LightGCN --loss DCL --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --tau_plus 0.1 --temp_tau 0.2\
            --comment _ --if_valid 0 --notes temp_tau=0.2 --tag DCL --group baseline --job_type yelp2018')#temp_tau=0.2
#Baseline--yelp2018--11 LightGCN + PD(Casaul BPR)
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+PD --model LightGCN --loss Causal_pop --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --pop_gamma 0.02\
            --comment _ --if_valid 0 --notes only_PD_gamma=0.02 --tag PDA --group baseline --job_type yelp2018')#Only PD, gamma=0.02


#Ours-----yelp2018--1 LightGCN + Adaptive_softmax_loss(log_pop_u, log_pop_i, centroid(eigenvector), commonNeighbor_u2i(SC), commonNeighbor_i2u(SC))
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+Adaptive_loss_mlp --model LightGCN --loss Adaptive --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --adaptive_method mlp --centroid_mode eigenvector --commonNeighbor_mode SC --temp_tau 0.1\
            --comment _ --if_valid 0 --notes LightGCN+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --tag Adaptive_loss --group Ours --job_type yelp2018')

#Ours-----yelp2018--2 SimGCL + Adaptive_softmax_loss(log_pop_u, log_pop_i, centroid(eigenvector), commonNeighbor_u2i(SC), commonNeighbor_i2u(SC))
os.system(f'python main.py --project GCLRec_No_Valid --name SimGCL+Adaptive_loss_mlp --model SimGCL --loss Adaptive --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --adaptive_method mlp --centroid_mode eigenvector --commonNeighbor_mode SC\
            --lambda1 0.5 --temp_tau 0.2 --eps_SimGCL 0.1 --augment No\
            --comment _ --if_valid 0 --notes SimGCL+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --tag Adaptive_loss --group Ours --job_type yelp2018')#two temp_tau, one in SimGCL's contrastive loss, another in Adaptive loss




















'''
#========================================GOWALLA========================================
#Baseline--gowalla--1 LightGCN + BPR
os.system(f'python main.py --project GCLRec --name LightGCN+BPR --model LightGCN --loss BPR --dataset gowalla\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 LightGCN_PyG + BPR
os.system(f'python main.py --project GCLRec --name LightGCN_PyG+BPR --model LightGCN_PyG --loss BPR --dataset gowalla\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_ED+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_ED+BPR_CL --model SGL --loss BPR_Contrast --augment ED --dataset gowalla\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_RW+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset gowalla\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0')

#========================================AMAZON-BOOK========================================
#Baseline--amazon-book--1 LightGCN + BPR
os.system(f'python main.py --project GCLRec --name LightGCN+BPR --model LightGCN --loss BPR --dataset amazon-book\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_ED+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_ED+BPR_CL --model SGL --loss BPR_Contrast --augment ED --dataset amazon-book\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.5 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_RW+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset amazon-book\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.5 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0')

#========================================IFASHION========================================
#Baseline--ifashion--1 LightGCN + BPR
os.system(f'python main.py --project GCLRec --name LightGCN+BPR --model LightGCN --loss BPR --dataset ifashion\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_ED+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_ED+BPR_CL --model SGL --loss BPR_Contrast --augment ED --dataset ifashion\
            --init_method Normal --lr 0.001 --weight_decay 1e-3 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.02 --p_drop 0.4 --temp_tau 0.5 \
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_RW+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset ifashion\
            --init_method Normal --lr 0.001 --weight_decay 1e-3 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.02 --p_drop 0.4 --temp_tau 0.5 \
            --comment _ --if_valid 0')


#========================================MIND========================================
#Baseline--MIND--1 LightGCN + BPR
os.system(f'python main.py --project GCLRec --name LightGCN+BPR --model LightGCN --loss BPR --dataset MIND\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_ED+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_ED+BPR_CL --model SGL --loss BPR_Contrast --augment ED --dataset MIND\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0')
#Baseline--yelp2018--1 SGL_RW+BPR_Contrast
os.system(f'python main.py --project GCLRec --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset MIND\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0')
'''