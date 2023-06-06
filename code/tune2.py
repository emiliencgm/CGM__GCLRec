import os
cuda = 1
# Baseline--yelp2018--2 LightGCN_PyG + BPR
os.system(f'python main.py --project GCLRec_No_Valid --name LightGCN_PyG+BPR --model LightGCN_PyG --loss BPR --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048\
            --comment _ --if_valid 0 --notes PyG_Implementation --tag LightGCN --group baseline --job_type yelp2018')

#Baseline--yelp2018--4 SGL_RW+BPR_Contrast
os.system(f'python main.py --project GCLRec_No_Valid --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 \
            --comment _ --if_valid 0 --notes _ --tag SGL --group baseline --job_type yelp2018')

#Baseline--yelp2018--6 SimGCL+BPR_Contrast
os.system(f'python main.py --project GCLRec_No_Valid --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset yelp2018\
            --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {cuda} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 \
            --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1\
            --comment _ --if_valid 0 --notes _ --tag SimGCL --group baseline --job_type yelp2018')#Layer=3