"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import dataloader
import precalcul
import world
from world import cprint
from world import cprint_rare
import model
import augment
import loss
import procedure
import torch
from tensorboardX import SummaryWriter
from os.path import join
import time
import visual
from pprint import pprint
import utils
from augment import Homophily
import wandb


def main():
    project = world.config['project']
    name = world.config['name']
    tag = world.config['tag']
    notes = world.config['notes']
    group = world.config['group']
    job_type = world.config['job_type']
    wandb.init(project=project, name=name, tags=tag, group=group, job_type=job_type, config=world.config, save_code=True, sync_tensorboard=False, notes=notes)
    wandb.define_metric("custom_epoch")
    wandb.define_metric(f"{world.config['dataset']}"+'/loss', step_metric='custom_epoch')
    for k in world.config['topks']:
        wandb.define_metric(f"{world.config['dataset']}"+f'/recall@{str(k)}', step_metric='custom_epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/ndcg@{str(k)}', step_metric='custom_epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/precision@{str(k)}', step_metric='custom_epoch')
        for group in range(world.config['pop_group']):
            wandb.define_metric(f"{world.config['dataset']}"+f"/groups/recall_group_{group+1}@{str(k)}", step_metric='custom_epoch')
    wandb.define_metric(f"{world.config['dataset']}"+f"/time_cost_s", step_metric='custom_epoch')


    world.make_print_to_file()

    utils.set_seed(world.config['seed'])

    print('==========config==========')
    pprint(world.config)
    print('==========config==========')

    cprint('[DATALOADER--START]')
    datasetpath = join(world.DATA_PATH, world.config['dataset'])
    dataset = dataloader.dataset(world.config, datasetpath)
    cprint('[DATALOADER--END]')

    cprint('[PRECALCULATE--START]')
    start = time.time()
    precal = precalcul.precalculate(world.config, dataset)
    end = time.time()
    print('precal cost : ',end-start)
    cprint('[PRECALCULATE--END]')

    models = {'LightGCN':model.LightGCN, 'GTN':model.GTN, 'SGL':model.SGL, 'SimGCL':model.SimGCL, 'GCLRec':model.GCLRec, 'LightGCN_PyG':model.LightGCN_PyG}
    Recmodel = models[world.config['model']](world.config, dataset, precal).to(world.device)

    homophily = Homophily(Recmodel)

    augments = {'No':None, 'ED':augment.ED_Uniform, 'RW':augment.RW_Uniform, 'SVD':augment.SVD_Augment, 'Adaptive':augment.Adaptive_Neighbor_Augment, 'Learner':augment.Augment_Learner}
    if world.config['augment'] in ['ED', 'RW', 'SVD', 'Adaptive']:
        augmentation = augments[world.config['augment']](world.config, Recmodel, precal, homophily)
    elif world.config['augment'] in ['Learner']:
        augmentation = augments[world.config['augment']](world.config, Recmodel, precal, homophily, dataset).to(world.device)
    else:
        augmentation = None

    losss = {'BPR': loss.BPR_loss, 'BPR_Contrast':loss.BPR_Contrast_loss, 'Softmax':loss.Softmax_loss, 'BC':loss.BC_loss, 'Adaptive':loss.Adaptive_softmax_loss, 'Causal_pop':loss.Causal_popularity_BPR_loss, 'DCL':loss.Debiased_Contrastive_loss}
    total_loss = losss[world.config['loss']](world.config, Recmodel, precal, homophily)

    w = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + str([(key,value)for key,value in world.log.items()])))

    train = procedure.Train(total_loss)
    test = procedure.Test()

    #TODO 检查全部待训练参数是否已经加入优化器
    optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
    if world.config['loss'] == 'Adaptive':
        optimizer.add_param_group({'params':total_loss.MLP_model.parameters()})
    if world.config['if_projector']:
        optimizer.add_param_group({'params':train.projector.parameters()})
    if world.config['augment'] in ['Learner']:
        optimizer.add_param_group({'params':augmentation.GNN_encoder.parameters()})
        optimizer.add_param_group({'params':augmentation.mlp_edge_model.parameters()})
    # optimizer = 0


    quantify = visual.Quantify(dataset, Recmodel, precal)


    try:
        best_result_recall = 0.
        best_result_ndcg = 0.
        stopping_step = 0
        if world.config['if_valid']:
            best_valid_recall = 0.
            stopping_valid_step = 0


        for epoch in range(world.config['epochs']):
            wandb.log({"custom_epoch": epoch})
            start = time.time()
            #====================VISUAL====================
            if world.config['if_visual'] == 1 and epoch % world.config['visual_epoch'] == 0:
                cprint("[Visualization]")
                if world.config['if_tsne'] == 1:
                    quantify.visualize_tsne(epoch)
                if world.config['if_double_label'] == 1:
                    quantify.visualize_double_label(epoch)
            #====================AUGMENT====================
            cprint('[AUGMENT]')
            if world.config['model'] in ['SGL']:
                augmentation.get_augAdjMatrix()

            
            #====================TRAIN====================
            cprint('[TRAIN]')
            start_train = time.time()
            # avg_loss = train.ForkMerge_train(dataset, Recmodel, augmentation, epoch, optimizer, w)
            avg_loss = train.train(dataset, Recmodel, augmentation, epoch, optimizer, w)
            end_train = time.time()
            wandb.log({ f"{world.config['dataset']}"+'/loss': avg_loss})
            wandb.log({f"{world.config['dataset']}"+f"/training_time": end_train - start_train})

            if epoch % 1== 0:
                #====================VALID====================
                if world.config['if_valid']:
                    cprint("[valid]")
                    result = test.valid(dataset, Recmodel, precal, epoch, w, world.config['if_multicore'])
                    if result["recall"][0] > best_valid_recall:#默认按照@20的效果early stop
                        stopping_valid_step = 0
                        advance = (result["recall"][0] - best_valid_recall)
                        best_valid_recall = result["recall"][0]
                        # print("find a better model")
                        cprint_rare("find a better valid recall", str(best_valid_recall), extra='++'+str(advance))
                        wandb.run.summary['best valid recall'] = best_valid_recall  
                    else:
                        stopping_valid_step += 1
                        if stopping_valid_step >= world.config['early_stop_steps']:
                            print(f"early stop triggerd at epoch {epoch}, best valid recall: {best_valid_recall}")
                            #将当前参数配置和获得的最佳结果记录
                            break
                    for i in range(len(world.config['topks'])):
                        k = world.config['topks'][i]
                        wandb.log({ f"{world.config['dataset']}"+f'/valid_recall@{str(k)}': result["recall"][i],
                                    f"{world.config['dataset']}"+f'/valid_ndcg@{str(k)}': result["ndcg"][i],
                                    f"{world.config['dataset']}"+f'/valid_precision@{str(k)}': result["precision"][i]})
                        
                #====================TEST====================
                cprint("[TEST]")
                result = test.test(dataset, Recmodel, precal, epoch, w, world.config['if_multicore'])
                if result["recall"][0] > best_result_recall:#默认按照@20的效果early stop
                    stopping_step = 0
                    advance = (result["recall"][0] - best_result_recall)
                    best_result_recall = result["recall"][0]
                    # print("find a better model")
                    cprint_rare("find a better recall", str(best_result_recall), extra='++'+str(advance))
                    wandb.run.summary['best test recall'] = best_result_recall  

                    # if world.config['if_visual'] == 1:
                    #     cprint("[Visualization]")
                    #     if world.config['if_tsne'] == 1:
                    #         quantify.visualize_tsne(epoch)
                    #     if world.config['if_double_label'] == 1:
                    #         quantify.visualize_double_label(epoch)

                    #torch.save(Recmodel.state_dict(), weight_file)
                else:
                    stopping_step += 1
                    if stopping_step >= world.config['early_stop_steps']:
                        print(f"early stop triggerd at epoch {epoch}, best recall: {best_result_recall}")
                        #将当前参数配置和获得的最佳结果记录
                        break
                for i in range(len(world.config['topks'])):
                    k = world.config['topks'][i]
                    wandb.log({ f"{world.config['dataset']}"+f'/recall@{str(k)}': result["recall"][i],
                                f"{world.config['dataset']}"+f'/ndcg@{str(k)}': result["ndcg"][i],
                                f"{world.config['dataset']}"+f'/precision@{str(k)}': result["precision"][i]})
                    for group in range(world.config['pop_group']):
                        wandb.log({f"{world.config['dataset']}"+f"/groups/recall_group_{group+1}@{str(k)}": result['recall_pop_Contribute'][group][i]})

            during = time.time() - start
            print(f"total time cost of epoch {epoch}: ", during)
    finally:
        w.close()
        wandb.finish()


if __name__ == '__main__':
    main()