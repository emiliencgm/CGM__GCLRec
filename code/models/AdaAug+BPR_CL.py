import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--task', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--L', type=int, default=3, help="total layers of LightGCN")
    parser.add_argument('--k', type=int, default=0, help="k-th layer to perturbate")
    parser.add_argument('--eps', type=float, default=0.1, help="epsilon_GCLRec")
    parser.add_argument('--w', type=float, default=0.1, help="w_GCLRec")
    return parser.parse_args()
args = parse_args()

#hyperparameters: InfoNCE:[temp_tau, lambda1, alpha], epsilon_GCLRec, w_GCLRec, k_aug, num_layers

if args.task == 'yelp2018':
    os.system(f'python main.py --notes L={str(args.L)}_k={str(args.k)} --alpha 0.5 --temp_tau 0.1 --lambda1 0.1 --epsilon_GCLRec {args.eps} --w_GCLRec {args.w} --k_aug {args.k} --project GCLRec_No_Valid --name Adaptive_Aug+BPR_Contrast --model GCLRec --loss BPR_Contrast --augment Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers {args.LGN_layer} --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag Adaptive_Aug --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')

elif args.task == 'gowalla':
    os.system(f'python main.py --notes L={str(args.L)}_k={str(args.k)} --alpha 0.5 --temp_tau 0.1 --lambda1 0.1 --epsilon_GCLRec {args.eps} --w_GCLRec {args.w} --k_aug {args.k} --project GCLRec_No_Valid --name Adaptive_Aug+BPR_Contrast --model GCLRec --loss BPR_Contrast --augment Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers {args.LGN_layer} --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag Adaptive_Aug --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')
    
elif args.task == 'amazon-book':
    os.system(f'python main.py --notes L={str(args.L)}_k={str(args.k)} --alpha 0.5 --temp_tau 0.1 --lambda1 0.1 --epsilon_GCLRec {args.eps} --w_GCLRec {args.w} --k_aug {args.k} --project GCLRec_No_Valid --name Adaptive_Aug+BPR_Contrast --model GCLRec --loss BPR_Contrast --augment Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers {args.LGN_layer} --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag Adaptive_Aug --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')

elif args.task == 'ifashion':
    os.system(f'python main.py --notes L={str(args.L)}_k={str(args.k)} --alpha 0.5 --temp_tau 0.1 --lambda1 0.1 --epsilon_GCLRec {args.eps} --w_GCLRec {args.w} --k_aug {args.k} --project GCLRec_No_Valid --name Adaptive_Aug+BPR_Contrast --model GCLRec --loss BPR_Contrast --augment Adaptive --dataset {args.task} --init_method Normal --lr 0.001 --weight_decay 1e-4 --cuda {args.device} --num_layers {args.LGN_layer} --latent_dim_rec 64 --batch_size 2048 --comment _ --if_valid 0 --tag Adaptive_Aug --group Ours --job_type {args.task} --if_visual {args.visual} --visual_epoch 1')