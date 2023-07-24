import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--valid', type=int, default=0, help="validation")
    return parser.parse_args()
args = parse_args()
if args.valid == 1:
    project = 'GCLRec_Valid'
else:
    project = 'GCLRec_No_Valid'

#hyperparameters: eps_SimGCL, temp_tau, lambda1

if args.task == 'yelp2018':
    os.system(f'python main.py --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1 --notes _ --project {project} --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag SimGCL --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')

elif args.task == 'gowalla':
    os.system(f'python main.py --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1 --notes _ --project {project} --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag SimGCL --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')
    
elif args.task == 'amazon-book':
    os.system(f'python main.py --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1 --notes _ --project {project} --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag SimGCL --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')

elif args.task == 'ifashion':
    os.system(f'python main.py --lambda1 0.1 --temp_tau 0.2 --eps_SimGCL 0.1 --notes _ --project {project} --name SimGCL+BPR_CL --model SimGCL --loss BPR_Contrast --augment No --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag SimGCL --group baseline --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')