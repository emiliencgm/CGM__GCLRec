import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--valid', type=int, default=0, help="validation")
    parser.add_argument('--tau', type=float, default=0.12, help="temp_tau")
    parser.add_argument('--centroid', type=str, default='eigenvector', help="centroid_mode")
    parser.add_argument('--CN', type=str, default='SC', help="commonNeighbor_mode")

    parser.add_argument('--mode', type=str, default='pos', help="Adaloss mode: [pos, pos+neg, pos+neg+cl]")
    parser.add_argument('--model', type=str, default='LightGCN', help="model for AdaLoss [LightGCN, LightGCN_PyG, GTN, SGL, SimGCL, GCLRec]")
    parser.add_argument('--augment', type=str, default='No', help="Augmentation for AdaLoss [SGL--ED/RW, SimGCL--No, SVD, Adaptive, Learner]")

    return parser.parse_args()
args = parse_args()
if args.valid == 1:
    project = 'GCLRec_Valid'
else:
    project = 'GCLRec_No_Valid'
#hyperparameters: temp_tau, [adaptive_method==mlp, centroid_mode==eigenvector (pagerank, degree), commonNeighbor_mode==SC (JS, CN, LHN)]

if args.task == 'yelp2018':
    if args.model=='SimGCL':
        os.system(f'python main.py --notes SimGCL+InfoNCE_+_AdaLoss_mlp --lambda1 0.1 --eps_SimGCL 0.1 --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name SimGCL+infoNCE+AdaLoss --model SimGCL --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='SGL-ED':
        os.system(f'python main.py --notes SGL_ED+InfoNCE_+_AdaLoss_mlp --lambda1 0.1 --eps_SimGCL 0.1 --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name SGL+infoNCE+AdaLoss --model SGL --augment ED --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='SGL-RW':
        os.system(f'python main.py --notes SGL_ED+InfoNCE_+_AdaLoss_mlp --lambda1 0.1 --eps_SimGCL 0.1 --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name SGL+infoNCE+AdaLoss --model SGL --augment RW --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='LightGCN':
        os.system(f'python main.py --notes LGN_+_AdaLoss_mlp --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name LightGCN+AdaLoss --model LightGCN --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='LightGCN_PyG':
        os.system(f'python main.py --notes LGN_PyG_+_AdaLoss_mlp --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name LightGCN_PyG+AdaLoss --model LightGCN_PyG --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='GTN':
        os.system(f'python main.py --notes GTN_+_AdaLoss_mlp --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name GTN+AdaLoss --model GTN --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='LightGCN-SVD':
        os.system(f'python main.py --notes LGN_+_SVD_+_AdaLoss_mlp --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name LightGCN+SVD+AdaLoss --model LightGCN --augment SVD --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='LightGCN-Learner':
        os.system(f'python main.py --notes LGN_+_Learner_+_AdaLoss_mlp --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name LightGCN+Learner+AdaLoss --model LightGCN --augment Learner --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    if args.model=='GCLRec':
        os.system(f'python main.py --notes GCLRec_+_AdaNeigh_+_AdaLoss_mlp --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name GCLRec+AdaNeigh+AdaLoss --model GCLRec --augment Adaptive --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    



elif args.task == 'gowalla':
    if args.model=='SimGCL':
        os.system(f'python main.py --notes SimGCL+InfoNCE_+_AdaLoss_mlp --lambda1 0.1 --eps_SimGCL 0.1 --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name SimGCL+infoNCE+AdaLoss --model SimGCL --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    
elif args.task == 'amazon-book':
    if args.model=='SimGCL':
        os.system(f'python main.py --notes SimGCL+InfoNCE_+_AdaLoss_mlp --lambda1 0.1 --eps_SimGCL 0.1 --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name SimGCL+infoNCE+AdaLoss --model SimGCL --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')
    
elif args.task == 'ifashion':
    if args.model=='SimGCL':
        os.system(f'python main.py --notes SimGCL+InfoNCE_+_AdaLoss_mlp --lambda1 0.1 --eps_SimGCL 0.1 --temp_tau {args.tau} --adaptive_method mlp --centroid_mode {args.centroid} --commonNeighbor_mode {args.CN} --project {project} --name SimGCL+infoNCE+AdaLoss --model SimGCL --augment No --loss Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers 3 --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid {args.valid} --tag Adaptive_loss --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 3')