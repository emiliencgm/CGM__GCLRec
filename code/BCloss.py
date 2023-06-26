import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--valid', type=int, default=0, help="validation")
    parser.add_argument('--temp_tau', type=float, default=0.1, help="temp_tau")
    return parser.parse_args()
args = parse_args()
if args.valid == 1:
    project = 'GCLRec_Valid'
else:
    project = 'GCLRec_No_Valid'
#hyperparameters: temp_tau, temp_tau_pop, alpha(weight between loss_1 and loss_pop_2)

if args.task == 'yelp2018':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop {args.temp_tau} --temp_tau {args.temp_tau} --notes tau1=tau2 --project {project} --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')

elif args.task == 'gowalla':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop {args.temp_tau} --temp_tau {args.temp_tau} --notes tau1=tau2 --project {project} --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')
    
elif args.task == 'amazon-book':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop {args.temp_tau} --temp_tau {args.temp_tau} --notes tau1=tau2 --project {project} --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')
    
elif args.task == 'ifashion':
    os.system(f'python main.py --alpha 0.5 --temp_tau_pop {args.temp_tau} --temp_tau {args.temp_tau} --notes tau1=tau2 --project {project} --name LightGCN+BC_loss --model LightGCN --loss BC --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag BC_loss --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 5')