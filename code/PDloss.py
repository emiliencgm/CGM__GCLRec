import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    return parser.parse_args()
args = parse_args()

#hyperparameters: pop_gamma

if args.task == 'yelp2018':
    os.system(f'python main.py --pop_gamma 0.02 --notes only_PD_gamma=0.02 --project GCLRec_No_Valid --name LightGCN+PD --model LightGCN --loss Causal_pop --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag PDA --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 4')

elif args.task == 'gowalla':
    os.system(f'python main.py --pop_gamma 0.02 --notes only_PD_gamma=0.02 --project GCLRec_No_Valid --name LightGCN+PD --model LightGCN --loss Causal_pop --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag PDA --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 4')

elif args.task == 'amazon-book':
    os.system(f'python main.py --pop_gamma 0.02 --notes only_PD_gamma=0.02 --project GCLRec_No_Valid --name LightGCN+PD --model LightGCN --loss Causal_pop --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag PDA --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 4')

elif args.task == 'ifashion':
    os.system(f'python main.py --pop_gamma 0.02 --notes only_PD_gamma=0.02 --project GCLRec_No_Valid --name LightGCN+PD --model LightGCN --loss Causal_pop --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag PDA --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 4')