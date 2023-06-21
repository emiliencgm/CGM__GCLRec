import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--valid', type=int, default=1, help="validation")
    parser.add_argument('--tau', type=float, default=0.12, help="temp_tau")
    parser.add_argument('--centroid', type=str, default='eigenvector', help="centroid_mode")
    parser.add_argument('--CN', type=str, default='SC', help="commonNeighbor_mode")

    return parser.parse_args()
args = parse_args()
if args.valid == 1:
    project = 'GCLRec_Valid'
else:
    project = 'GCLRec_No_Valid'
#hyperparameters: temp_tau, [adaptive_method==mlp, centroid_mode==eigenvector (pagerank, degree), commonNeighbor_mode==SC (JS, CN, LHN)]

if args.task == 'yelp2018':
    os.system(f'python main.py --notes LightGCN+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --temp_tau {args.tau} --project {project} --name LightGCN+Adaptive_loss_mlp --model LightGCN --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')

elif args.task == 'gowalla':
    os.system(f'python main.py --notes LightGCN+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --temp_tau {args.tau} --project {project} --name LightGCN+Adaptive_loss_mlp --model LightGCN --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    
elif args.task == 'amazon-book':
    os.system(f'python main.py --notes LightGCN+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --temp_tau {args.tau} --project {project} --name LightGCN+Adaptive_loss_mlp --model LightGCN --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    
elif args.task == 'ifashion':
    os.system(f'python main.py --notes LightGCN+Adaptive_loss__mlp__log_pop+centroid_eig+CN_SC__ --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --temp_tau {args.tau} --project {project} --name LightGCN+Adaptive_loss_mlp --model LightGCN --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')