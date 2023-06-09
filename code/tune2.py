import os
cuda = 0
# #Adaptive loss
# os.system('python main.py --project GCLRec_No_Valid --name LightGCN+Adaptive_loss_mlp --model LightGCN --loss Adaptive --dataset yelp2018 --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda 0 --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --adaptive_method mlp --centroid_mode eigenvector --commonNeighbor_mode SC --temp_tau 0.1 --comment _ --if_valid 0 --notes LightGCN+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --tag Adaptive_loss --group Ours --job_type yelp2018 --if_visual 1 --visual_epoch 3')

# #Adaptive Aug
# os.system('python main.py --project GCLRec_No_Valid --name Adaptive_Aug+BPR_Contrast --model GCLRec --loss BPR_Contrast --augment Adaptive --dataset yelp2018 --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda 0 --num_layers 4 --latent_dim_rec 64 --batch_size 2048 --alpha 0.5 --temp_tau 0.1 --lambda1 0.1 --epsilon_GCLRec 0.1 --w_GCLRec 0. --k_aug 1 --comment _ --if_valid 0 --notes just_neighbor_No_Adaptive_4Layers --tag Adaptive_Aug --group Ours --job_type yelp2018 --if_visual 1 --visual_epoch 1')


os.system('python main.py --project GCLRec_No_Valid --name Learner_Aug+BPR_Contrast --model LightGCN_PyG --loss BPR_Contrast --augment Learner --dataset yelp2018 --init_method Normal --adaptive_method None --temp_tau 0.1 --if_visual 0 --cuda 0 --comment tune_Aument_Learner_LGN_Contrast --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --early_stop_steps 30 --if_projector 0 --if_valid 0 --notes Aug_Learner_BPR_CL_tau=0.1 --tag Learner_Aug --group Ours --job_type yelp2018')