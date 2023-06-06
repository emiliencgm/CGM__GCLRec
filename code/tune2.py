import os
cuda = 0
# Baseline--yelp2018--2 LightGCN_PyG + BPR
# os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN_PyG+BPR --model LightGCN_PyG --loss BPR --augment No --dataset yelp2018\
#             --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
#             --comment _ --if_valid 0 --notes PyG_Implementation --tag LightGCN --group baseline --job_type yelp2018')

# #Baseline--yelp2018--4 SGL_RW+BPR_Contrast
# os.system(f'python main.py --project GCLRec_No_Valid --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset yelp2018\
#             --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
#             --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
#             --comment _ --if_valid 0 --notes _ --tag SGL --group baseline --job_type yelp2018')

#Baseline--yelp2018--6 SimGCL+BPR_Contrast
# os.system(f'python main.py --project GCLRec_No_Valid --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset yelp2018\
#             --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
#             --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1\
#             --comment _ --if_valid 0 --notes _ --tag SimGCL --group baseline --job_type yelp2018')#Layer=3

#Aument_Learner and LightGCN_PyG
# os.system(f'python main.py --model LightGCN_PyG --loss BPR_Contrast --augment Learner --dataset yelp2018\
#             --init_method Normal --adaptive_method None\
#             --temp_tau 0.2\
#             --if_visual 0 --cuda {cuda} --comment tune_Aument_Learner_LGN_Contrast\
#             --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
#             --early_stop_steps 40\
#             --if_projector 0\
#             --if_valid 0')
#Baseline--yelp2018--10 LightGCN + DCL
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+DCL --model LightGCN --loss DCL --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --tau_plus 0.1 --temp_tau 0.2\
            --comment _ --if_valid 0 --notes temp_tau=0.2 --tag DCL --group baseline --job_type yelp2018')#temp_tau=0.2
#Baseline--yelp2018--10 LightGCN + DCL
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN+DCL --model LightGCN --loss DCL --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --tau_plus 0.1 --temp_tau 0.1\
            --comment _ --if_valid 0 --notes temp_tau=0.1 --tag DCL --group baseline --job_type yelp2018')#temp_tau=0.1