import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    return parser.parse_args()
args = parse_args()

#hyperparameters: temp_tau, temp_tau_pop, alpha(weight between loss_1 and loss_pop_2)

if args.task == 'yelp2018':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.1 --notes tau1=tau2=0.1 --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')

elif args.task == 'gowalla':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.1 --notes tau1=tau2=0.1 --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')
    
elif args.task == 'amazon-book':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.1 --notes tau1=tau2=0.1 --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')
    
elif args.task == 'ifashion':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop 0.1 --temp_tau 0.1 --notes tau1=tau2=0.1 --project GCLRec_No_Valid --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')