import os
#BC_loss
os.system('python main.py --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset yelp2018 --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda 0 --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.1 --comment _ --if_valid 0 --notes tau1=tau2=0.1 --tag BC_loss --group baseline --job_type yelp2018 --if_visual 1 --visual_epoch 5')

#SimGCL
os.system('python main.py --project GCLRec_No_Valid --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset yelp2018 --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda 0 --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1 --comment _ --if_valid 0 --notes _ --tag SimGCL --group baseline --job_type yelp2018 --if_visual 1 --visual_epoch 1')

#SGL_RW
os.system('python main.py --project GCLRec_No_Valid --name SGL_RW+BPR_CL --model SGL --loss BPR_Contrast --augment RW --dataset yelp2018 --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda 0 --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 --comment _ --if_valid 0 --notes _ --tag SGL --group baseline --job_type yelp2018 --if_visual 1 --visual_epoch 1')

#SGL_ED
os.system('python main.py --project GCLRec_No_Valid --name SGL_ED+BPR_CL --model SGL --loss BPR_Contrast --augment ED --dataset yelp2018 --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda 0 --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --lambda1 0.1 --p_drop 0.1 --temp_tau 0.2 --comment _ --if_valid 0 --notes _ --tag SGL --group baseline --job_type yelp2018 --if_visual 1 --visual_epoch 1')