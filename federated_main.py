import argparse
import torch
from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time

import os
import gc
import copy
from prettytable import PrettyTable
import numpy as np
from utils.fed_utils import average_weights, count_parameters

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.peft:
        cfg.PEFT = args.peft
    
    if args.tau:
        cfg.TAU = args.tau
    
    if args.lora_encoder:
        cfg.LORA.ENCODER = args.lora_encoder
    
    if args.lora_rank:
        cfg.LORA.R = args.lora_rank


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    # Config for FLORA
    cfg.TRAINER.FLORA = CN()
    cfg.TRAINER.FLORA.PREC = "fp16"  # fp16, fp32, amp


    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain # number of domain
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = args.split_client # is adding label skew to feature skew datasets and split one domain to multi clients
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = 1 # local epoch
    cfg.OPTIM.GAMMA = args.gamma # gamma of single-step
    cfg.OPTIM.LR = args.lr #learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
   
    local_weights= [[] for i in range(args.num_users)]
    local_weights_0= [[] for i in range(args.num_users)]
    local_weights_1= [[] for i in range(args.num_users)]
    local_weights_2= [[] for i in range(args.num_users)]
    local_weights_per = [{} for i in range(args.num_users)]
    local_proj = [{} for i in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model,"prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")

    # local_trainers = {net_i: None for net_i in range(cfg.DATASET.USERS)}
    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))
            # local_trainer.fed_before_train()
            # local_trainers[net_i] = local_trainer
            # local_weights[net_i] = copy.deepcopy(local_trainer.model.state_dict())
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_test_ece_list = []
    global_test_mce_list = []
    global_test_ace_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()
    n_cls = len(local_trainer.dm.dataset.classnames)
    for epoch in range(start_epoch, max_epoch):

        if args.trainer == 'CLIP':
            print("dataset_name", cfg.DATASET.NAME)
            print("------------local test start-------------")
            results = []
            # idxs_users = list(range(0,cfg.DATASET.USERS))
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights)
                results.append(local_trainer.test(idx=idx))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            global_test_ece = []
            global_test_mce = []
            global_test_ace = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
                global_test_ece.append(results[k][4])
                global_test_mce.append(results[k][5])
                global_test_ace.append(results[k][6])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_test_ece_list.append(sum(global_test_ece) / len(global_test_ece))
            global_test_mce_list.append(sum(global_test_mce) / len(global_test_mce))
            global_test_ace_list.append(sum(global_test_ace) / len(global_test_ace))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            print("Global test ece:", sum(global_test_ece) / len(global_test_ece))
            print("Global test mce:", sum(global_test_mce) / len(global_test_mce))
            print("Global test ace:", sum(global_test_ace) / len(global_test_ace))

            if cfg.DATASET.NAME == 'PACS':
                if epoch == max_epoch-1:
                    print("Test acc of clients:", global_test_acc)
                    print("Test ece of clients:", global_test_ece)
                    print("Test mce of clients:", global_test_mce)
                    print("Test ace of clients:", global_test_ace)
                    
                    print("Test acc of art painting", np.mean(global_test_acc[0:2]),"±",np.std(global_test_acc[0:2]))
                    print("Test ece of art painting", np.mean(global_test_ece[0:2]),"±",np.std(global_test_ece[0:2]))
                    print("Test mce of art painting", np.mean(global_test_mce[0:2]),"±",np.std(global_test_mce[0:2]))
                    print("Test ace of art painting", np.mean(global_test_ace[0:2]),"±",np.std(global_test_ace[0:2]))

                    print("Test acc of cartoon", np.mean(global_test_acc[2:4]),"±",np.std(global_test_acc[2:4]))
                    print("Test ece of cartoon", np.mean(global_test_ece[2:4]),"±",np.std(global_test_ece[2:4]))
                    print("Test mce of cartoon", np.mean(global_test_mce[2:4]),"±",np.std(global_test_mce[2:4]))
                    print("Test ace of cartoon", np.mean(global_test_ace[2:4]),"±",np.std(global_test_ace[2:4]))

                    print("Test acc of photo", np.mean(global_test_acc[4:6]),"±",np.std(global_test_acc[4:6]))
                    print("Test ece of photo", np.mean(global_test_ece[4:6]),"±",np.std(global_test_ece[4:6]))
                    print("Test mce of photo", np.mean(global_test_mce[4:6]),"±",np.std(global_test_mce[4:6]))
                    print("Test ace of photo", np.mean(global_test_ace[4:6]),"±",np.std(global_test_ace[4:6]))

                    print("Test acc of sketch", np.mean(global_test_acc[6:8]),"±",np.std(global_test_acc[6:8]))
                    print("Test ece of sketch", np.mean(global_test_ece[6:8]),"±",np.std(global_test_ece[6:8]))
                    print("Test mce of sketch", np.mean(global_test_mce[6:8]),"±",np.std(global_test_mce[6:8]))
                    print("Test ace of sketch", np.mean(global_test_ace[6:8]),"±",np.std(global_test_ace[6:8]))

                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                    print("Test ece of all",np.mean(global_test_ece),np.std(global_test_ece))
                    print("Test mce of all",np.mean(global_test_mce),np.std(global_test_mce))
                    print("Test ace of all",np.mean(global_test_ace),np.std(global_test_ace))
            
            elif cfg.DATASET.NAME == 'OfficeHome':
                if epoch == max_epoch-1:
                    print("Test acc of clients:", global_test_acc)
                    print("Test ece of clients:", global_test_ece)
                    print("Test mce of clients:", global_test_mce)
                    print("Test ace of clients:", global_test_ace)
                    
                    print("Test acc of art", np.mean(global_test_acc[0:2]),"±",np.std(global_test_acc[0:2]))
                    print("Test ece of art", np.mean(global_test_ece[0:2]),"±",np.std(global_test_ece[0:2]))
                    print("Test mce of art", np.mean(global_test_mce[0:2]),"±",np.std(global_test_mce[0:2]))
                    print("Test ace of art", np.mean(global_test_ace[0:2]),"±",np.std(global_test_ace[0:2]))

                    print("Test acc of clipart", np.mean(global_test_acc[2:4]),"±",np.std(global_test_acc[2:4]))
                    print("Test ece of clipart", np.mean(global_test_ece[2:4]),"±",np.std(global_test_ece[2:4]))
                    print("Test mce of clipart", np.mean(global_test_mce[2:4]),"±",np.std(global_test_mce[2:4]))
                    print("Test ace of clipart", np.mean(global_test_ace[2:4]),"±",np.std(global_test_ace[2:4]))

                    print("Test acc of product", np.mean(global_test_acc[4:6]),"±",np.std(global_test_acc[4:6]))
                    print("Test ece of product", np.mean(global_test_ece[4:6]),"±",np.std(global_test_ece[4:6]))
                    print("Test mce of product", np.mean(global_test_mce[4:6]),"±",np.std(global_test_mce[4:6]))
                    print("Test ace of product", np.mean(global_test_ace[4:6]),"±",np.std(global_test_ace[4:6]))

                    print("Test acc of realworld", np.mean(global_test_acc[6:8]),"±",np.std(global_test_acc[6:8]))
                    print("Test ece of realworld", np.mean(global_test_ece[6:8]),"±",np.std(global_test_ece[6:8]))
                    print("Test mce of realworld", np.mean(global_test_mce[6:8]),"±",np.std(global_test_mce[6:8]))
                    print("Test ace of realworld", np.mean(global_test_ace[6:8]),"±",np.std(global_test_ace[6:8]))

                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                    print("Test ece of all",np.mean(global_test_ece),np.std(global_test_ece))
                    print("Test mce of all",np.mean(global_test_mce),np.std(global_test_mce))
                    print("Test ace of all",np.mean(global_test_ace),np.std(global_test_ace))

            elif cfg.DATASET.NAME == 'VLCS':
                if epoch == max_epoch-1:
                    print("Test acc of clients:", global_test_acc)
                    print("Test ece of clients:", global_test_ece)
                    print("Test mce of clients:", global_test_mce)
                    print("Test ace of clients:", global_test_ace)
                    
                    print("Test acc of CALTECH", np.mean(global_test_acc[0:2]),"±",np.std(global_test_acc[0:2]))
                    print("Test ece of CALTECH", np.mean(global_test_ece[0:2]),"±",np.std(global_test_ece[0:2]))
                    print("Test mce of CALTECH", np.mean(global_test_mce[0:2]),"±",np.std(global_test_mce[0:2]))
                    print("Test ace of CALTECH", np.mean(global_test_ace[0:2]),"±",np.std(global_test_ace[0:2]))

                    print("Test acc of LABELME", np.mean(global_test_acc[2:4]),"±",np.std(global_test_acc[2:4]))
                    print("Test ece of LABELME", np.mean(global_test_ece[2:4]),"±",np.std(global_test_ece[2:4]))
                    print("Test mce of LABELME", np.mean(global_test_mce[2:4]),"±",np.std(global_test_mce[2:4]))
                    print("Test ace of LABELME", np.mean(global_test_ace[2:4]),"±",np.std(global_test_ace[2:4]))

                    print("Test acc of PASCAL", np.mean(global_test_acc[4:6]),"±",np.std(global_test_acc[4:6]))
                    print("Test ece of PASCAL", np.mean(global_test_ece[4:6]),"±",np.std(global_test_ece[4:6]))
                    print("Test mce of PASCAL", np.mean(global_test_mce[4:6]),"±",np.std(global_test_mce[4:6]))
                    print("Test ace of PASCAL", np.mean(global_test_ace[4:6]),"±",np.std(global_test_ace[4:6]))

                    print("Test acc of SUN", np.mean(global_test_acc[6:8]),"±",np.std(global_test_acc[6:8]))
                    print("Test ece of SUN", np.mean(global_test_ece[6:8]),"±",np.std(global_test_ece[6:8]))
                    print("Test mce of SUN", np.mean(global_test_mce[6:8]),"±",np.std(global_test_mce[6:8]))
                    print("Test ace of SUN", np.mean(global_test_ace[6:8]),"±",np.std(global_test_ace[6:8]))

                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                    print("Test ece of all",np.mean(global_test_ece),np.std(global_test_ece))
                    print("Test mce of all",np.mean(global_test_mce),np.std(global_test_mce))
                    print("Test ace of all",np.mean(global_test_ace),np.std(global_test_ace))


            print("------------local test finish-------------")
            print("Epoch on server :", epoch)
            break
                
        elif args.model == 'FLORA':
            if epoch == 0:
                idxs_users = list(range(0,cfg.DATASET.USERS))
            else:              
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights,strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx],strict=False)
    
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy({name: param for name, param in local_weight.items() if cfg.PEFT in name})

            print("------------local train finish epoch:", epoch, "-------------")
            global_weights = average_weights(local_weights_0,idxs_users, datanumber_client,islist=False)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0,cfg.DATASET.USERS))
            
            for idx in all_users:
                for key in local_weights_per[idx].keys():
                    if key in global_weights:
                        local_weights_per[idx][key] = global_weights[key]               
    
            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx],strict=False)
                results.append(local_trainer.test(idx=idx))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            global_test_ece = []
            global_test_mce = []
            global_test_ace = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
                global_test_ece.append(results[k][4])
                global_test_mce.append(results[k][5])
                global_test_ace.append(results[k][6])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_test_ece_list.append(sum(global_test_ece) / len(global_test_ece))
            global_test_mce_list.append(sum(global_test_mce) / len(global_test_mce))
            global_test_ace_list.append(sum(global_test_ace) / len(global_test_ace))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            print("Global test ece:", sum(global_test_ece) / len(global_test_ece))
            print("Global test mce:", sum(global_test_mce) / len(global_test_mce))
            print("Global test ace:", sum(global_test_ace) / len(global_test_ace))

            if cfg.DATASET.NAME == 'PACS':
                if epoch == max_epoch-1:
                    print("Test acc of clients:", global_test_acc)
                    print("Test ece of clients:", global_test_ece)
                    print("Test mce of clients:", global_test_mce)
                    print("Test ace of clients:", global_test_ace)
                    
                    print("Test acc of art painting", np.mean(global_test_acc[0:2]),"±",np.std(global_test_acc[0:2]))
                    print("Test ece of art painting", np.mean(global_test_ece[0:2]),"±",np.std(global_test_ece[0:2]))
                    print("Test mce of art painting", np.mean(global_test_mce[0:2]),"±",np.std(global_test_mce[0:2]))
                    print("Test ace of art painting", np.mean(global_test_ace[0:2]),"±",np.std(global_test_ace[0:2]))

                    print("Test acc of cartoon", np.mean(global_test_acc[2:4]),"±",np.std(global_test_acc[2:4]))
                    print("Test ece of cartoon", np.mean(global_test_ece[2:4]),"±",np.std(global_test_ece[2:4]))
                    print("Test mce of cartoon", np.mean(global_test_mce[2:4]),"±",np.std(global_test_mce[2:4]))
                    print("Test ace of cartoon", np.mean(global_test_ace[2:4]),"±",np.std(global_test_ace[2:4]))

                    print("Test acc of photo", np.mean(global_test_acc[4:6]),"±",np.std(global_test_acc[4:6]))
                    print("Test ece of photo", np.mean(global_test_ece[4:6]),"±",np.std(global_test_ece[4:6]))
                    print("Test mce of photo", np.mean(global_test_mce[4:6]),"±",np.std(global_test_mce[4:6]))
                    print("Test ace of photo", np.mean(global_test_ace[4:6]),"±",np.std(global_test_ace[4:6]))

                    print("Test acc of sketch", np.mean(global_test_acc[6:8]),"±",np.std(global_test_acc[6:8]))
                    print("Test ece of sketch", np.mean(global_test_ece[6:8]),"±",np.std(global_test_ece[6:8]))
                    print("Test mce of sketch", np.mean(global_test_mce[6:8]),"±",np.std(global_test_mce[6:8]))
                    print("Test ace of sketch", np.mean(global_test_ace[6:8]),"±",np.std(global_test_ace[6:8]))

                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                    print("Test ece of all",np.mean(global_test_ece),np.std(global_test_ece))
                    print("Test mce of all",np.mean(global_test_mce),np.std(global_test_mce))
                    print("Test ace of all",np.mean(global_test_ace),np.std(global_test_ace))
            
            elif cfg.DATASET.NAME == 'OfficeHome':
                if epoch == max_epoch-1:
                    print("Test acc of clients:", global_test_acc)
                    print("Test ece of clients:", global_test_ece)
                    print("Test mce of clients:", global_test_mce)
                    print("Test ace of clients:", global_test_ace)
                    
                    print("Test acc of art", np.mean(global_test_acc[0:2]),"±",np.std(global_test_acc[0:2]))
                    print("Test ece of art", np.mean(global_test_ece[0:2]),"±",np.std(global_test_ece[0:2]))
                    print("Test mce of art", np.mean(global_test_mce[0:2]),"±",np.std(global_test_mce[0:2]))
                    print("Test ace of art", np.mean(global_test_ace[0:2]),"±",np.std(global_test_ace[0:2]))

                    print("Test acc of clipart", np.mean(global_test_acc[2:4]),"±",np.std(global_test_acc[2:4]))
                    print("Test ece of clipart", np.mean(global_test_ece[2:4]),"±",np.std(global_test_ece[2:4]))
                    print("Test mce of clipart", np.mean(global_test_mce[2:4]),"±",np.std(global_test_mce[2:4]))
                    print("Test ace of clipart", np.mean(global_test_ace[2:4]),"±",np.std(global_test_ace[2:4]))

                    print("Test acc of product", np.mean(global_test_acc[4:6]),"±",np.std(global_test_acc[4:6]))
                    print("Test ece of product", np.mean(global_test_ece[4:6]),"±",np.std(global_test_ece[4:6]))
                    print("Test mce of product", np.mean(global_test_mce[4:6]),"±",np.std(global_test_mce[4:6]))
                    print("Test ace of product", np.mean(global_test_ace[4:6]),"±",np.std(global_test_ace[4:6]))

                    print("Test acc of realworld", np.mean(global_test_acc[6:8]),"±",np.std(global_test_acc[6:8]))
                    print("Test ece of realworld", np.mean(global_test_ece[6:8]),"±",np.std(global_test_ece[6:8]))
                    print("Test mce of realworld", np.mean(global_test_mce[6:8]),"±",np.std(global_test_mce[6:8]))
                    print("Test ace of realworld", np.mean(global_test_ace[6:8]),"±",np.std(global_test_ace[6:8]))

                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                    print("Test ece of all",np.mean(global_test_ece),np.std(global_test_ece))
                    print("Test mce of all",np.mean(global_test_mce),np.std(global_test_mce))
                    print("Test ace of all",np.mean(global_test_ace),np.std(global_test_ace))

            elif cfg.DATASET.NAME == 'VLCS':
                if epoch == max_epoch-1:
                    print("Test acc of clients:", global_test_acc)
                    print("Test ece of clients:", global_test_ece)
                    print("Test mce of clients:", global_test_mce)
                    print("Test ace of clients:", global_test_ace)
                    
                    print("Test acc of CALTECH", np.mean(global_test_acc[0:2]),"±",np.std(global_test_acc[0:2]))
                    print("Test ece of CALTECH", np.mean(global_test_ece[0:2]),"±",np.std(global_test_ece[0:2]))
                    print("Test mce of CALTECH", np.mean(global_test_mce[0:2]),"±",np.std(global_test_mce[0:2]))
                    print("Test ace of CALTECH", np.mean(global_test_ace[0:2]),"±",np.std(global_test_ace[0:2]))

                    print("Test acc of LABELME", np.mean(global_test_acc[2:4]),"±",np.std(global_test_acc[2:4]))
                    print("Test ece of LABELME", np.mean(global_test_ece[2:4]),"±",np.std(global_test_ece[2:4]))
                    print("Test mce of LABELME", np.mean(global_test_mce[2:4]),"±",np.std(global_test_mce[2:4]))
                    print("Test ace of LABELME", np.mean(global_test_ace[2:4]),"±",np.std(global_test_ace[2:4]))

                    print("Test acc of PASCAL", np.mean(global_test_acc[4:6]),"±",np.std(global_test_acc[4:6]))
                    print("Test ece of PASCAL", np.mean(global_test_ece[4:6]),"±",np.std(global_test_ece[4:6]))
                    print("Test mce of PASCAL", np.mean(global_test_mce[4:6]),"±",np.std(global_test_mce[4:6]))
                    print("Test ace of PASCAL", np.mean(global_test_ace[4:6]),"±",np.std(global_test_ace[4:6]))

                    print("Test acc of SUN", np.mean(global_test_acc[6:8]),"±",np.std(global_test_acc[6:8]))
                    print("Test ece of SUN", np.mean(global_test_ece[6:8]),"±",np.std(global_test_ece[6:8]))
                    print("Test mce of SUN", np.mean(global_test_mce[6:8]),"±",np.std(global_test_mce[6:8]))
                    print("Test ace of SUN", np.mean(global_test_ace[6:8]),"±",np.std(global_test_ace[6:8]))

                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                    print("Test ece of all",np.mean(global_test_ece),np.std(global_test_ece))
                    print("Test mce of all",np.mean(global_test_mce),np.std(global_test_mce))
                    print("Test ace of all",np.mean(global_test_ace),np.std(global_test_ace))

            print("------------local test finish-------------")
            print("Epoch on server :", epoch)


    for idx in idxs_users:
        local_trainer.fed_after_train()
    print("global_test_acc_list:",global_test_acc_list)
    print("maximum test acc:", max(global_test_acc_list))
    print("mean of acc:",np.mean(global_test_acc_list[-5:]))
    print("std of acc:",np.std(global_test_acc_list[-5:]))
    
    print("global_test_ece_list:",global_test_ece_list)
    print("minimum test ece:", min(global_test_ece_list))
    print("mean of ece:",np.mean(global_test_ece_list[-5:]))
    print("std of ece:",np.std(global_test_ece_list[-5:]))
    
    print("global_test_mce_list:",global_test_mce_list)
    print("minimum test mce:", min(global_test_mce_list))
    print("mean of mce:",np.mean(global_test_mce_list[-5:]))
    print("std of mce:",np.std(global_test_mce_list[-5:]))
    
    print("global_test_ace_list:",global_test_ace_list)
    print("minimum test ace:", min(global_test_ace_list))
    print("mean of ace:",np.mean(global_test_ace_list[-5:]))
    print("std of ace:",np.std(global_test_ace_list[-5:]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FLORA", help="model of aggregation")
    parser.add_argument("--trainer", type=str, default="FLORA", help="name of trainer")
    parser.add_argument('--round', type=int, default=50, help="number of communication round")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=128, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--peft", type=str, default="ln", help="PEFT method")
    parser.add_argument("--lora_encoder", type=str, default="both", help="LoRA encoder")
    parser.add_argument("--lora_rank", type=int, default=2, help="LoRA rank")
    parser.add_argument('--tau', type=float, default=1.0, help="temperature scaling")

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num_shots', type=int, default=2, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir100',help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.1,help='The parameter for the dirichlet distribution for data partitioning')
    # domainnet, office
    parser.add_argument('--imbalance_train', default=False, help="is adding label skew to feature skew datasets")
    parser.add_argument('--split_client', default=False, help="is adding label skew to feature skew datasets and split one domain to multi clients")
    parser.add_argument('--num_domain', type=int, default=4, help="number of domain")

    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="/DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/..", help="output directory")
    parser.add_argument("--config-file", type=str, default="configs/trainers/GLP_OT/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/caltech101.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()
    main(args)








