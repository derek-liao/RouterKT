import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SimCLRDatasetWrapper,
    MKMDatasetWrapper,
    get_diff_df,
    CounterDatasetWrapper
)
from models.akt import AKT
from models.cl4kt import CL4KT
from models.simplekt import SimpleKT
from models.RouterSimpleKT import RouterSimpleKT
from models.RouterAKT import RouterAKT
from models.RouterCL4KT import RouterCL4KT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
from stat_data import get_stat
import wandb
import time
from time import localtime
import statistics
import json
import random
import matplotlib.pyplot as plt
import glob
import csv
import sys  # Add sys import

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

def validate_head_config(num_attn_heads, num_shared_heads, num_selected_heads):
    """Validate router head configurations."""
    if num_attn_heads < num_shared_heads:
        raise ValueError(f"Total attention heads ({num_attn_heads}) must be greater than shared heads ({num_shared_heads})")
    if num_selected_heads > (num_attn_heads - num_shared_heads):
        raise ValueError(f"Selected heads ({num_selected_heads}) cannot exceed number of dynamic heads ({num_attn_heads - num_shared_heads})")

def process_head_weights(layer):
    """Process attention head weights for a given layer."""
    n_shared = layer.h_shared
    n_dynamic = layer.h - n_shared

    # Shared weights are always 1.0
    shared_weights = np.ones(n_shared)

    # Get dynamic weights
    dynamic_scores = layer.dynamic_scores
    dynamic_weights = dynamic_scores.mean(dim=0).detach().cpu().numpy()

    # Calculate std across batch and sequence dimensions for each head
    weights_std = np.zeros(n_shared + n_dynamic)
    weights_std[n_shared:] = dynamic_scores.std(dim=0).detach().cpu().numpy()

    return {
        'shared_weights': shared_weights,
        'dynamic_weights': dynamic_weights,
        'weights_std': weights_std
    }

def process_encoder_weights(encoder):
    """Process weights for an encoder's first and last layers."""
    if len(encoder) == 0:
        return None, None

    first_layer = encoder[0].attn
    last_layer = encoder[-1].attn

    first_layer_weights = process_head_weights(first_layer)
    last_layer_weights = process_head_weights(last_layer)

    return first_layer_weights, last_layer_weights

def main(config):
    # Initialize wandb
    if config.use_wandb:
        # Initialize wandb
        wandb.init(project="RouterKT-发布测试", entity="ringotc")
        
        # Set run name to include key parameters
        if hasattr(config.routerakt_config, 'routing_mode'):
            wandb.run.name = f"{config.data_name}_{config.model_name}_rm_{config.routerakt_config.routing_mode}_sh_{config.routerakt_config.num_selected_heads}_blw_{config.routerakt_config.balance_loss_weight}_regl_{config.routerakt_config.l2}"
            wandb.run.save()

    tm = localtime(time.time())
    params_str = f'{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}:{tm.tm_min}:{tm.tm_sec}'

    accelerator = Accelerator()
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")

    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    seed = train_config.seed
    set_seed(seed)

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len
    diff_order = train_config.diff_order
    diff_as_loss_weight = train_config.diff_as_loss_weight
    uniform = train_config.uniform
    describe = train_config.describe

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    test_result = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    df = pd.read_csv(df_path, sep="\t")

    users = df["user_id"].unique()
    np.random.shuffle(users)
    get_stat(data_name, df)
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    print("MODEL", model_name)
    print(dataset)

    # 创建保存模型和可视化的目录
    dir_name = os.path.join("saved_model", model_name, data_name, params_str)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
        # Remove experiment-specific condition
        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)

        valid_users = train_users[offset:]
        train_users = train_users[:offset]
        test_users = users[test_ids]

        df = get_diff_df(df, seq_len, num_skills, num_questions, total_cnt_init=config.total_cnt_init, diff_unk=config.diff_unk)
        train_df = df[df["user_id"].isin(train_users)]

        train_quantiles = None
        train_bincounts = None
        token_num = int(args.de_type.split('_')[1])
        boundaries = np.linspace(0, 1, num=token_num+1)
        train_quantiles = torch.Tensor([train_df['skill_diff'].quantile(i) for i in boundaries])
        if uniform:
            boundaries = torch.Tensor(boundaries)
            train_diff_buckets = torch.bucketize(torch.Tensor(train_df['skill_diff'].to_numpy()), boundaries)
            diff_quantiles = boundaries
        else:
            train_diff_buckets = torch.bucketize(torch.Tensor(train_df['skill_diff'].to_numpy()), train_quantiles)
            diff_quantiles = train_quantiles
        train_bincounts = torch.bincount(train_diff_buckets)

        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]

        train_dataset = dataset(train_df, seq_len, num_skills, num_questions, diff_df= train_df, diff_quantiles=diff_quantiles, name="train")
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions, diff_df= train_df, diff_quantiles=diff_quantiles, name="valid")
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions, diff_df= train_df, diff_quantiles=diff_quantiles, name="test")

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))

        if model_name == "routersimplekt":
            model_config = config.routersimplekt_config
            # Validate head configuration
            validate_head_config(
                model_config.num_attn_heads,
                model_config.num_shared_heads,
                model_config.num_selected_heads
            )
            model = RouterSimpleKT(device, num_skills, num_questions, seq_len, **model_config)
        elif model_name == "simplekt":
            model_config = config.simplekt_config
            model = SimpleKT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
        elif model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == "akt":
            model_config = config.akt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = AKT(device, num_skills, num_questions, seq_len, train_bincounts, **model_config)
        elif model_name == "routerakt":
            model_config = config.routerakt_config
            # Validate head configurtion

            model = RouterAKT(
                device,
                num_skills,
                num_questions,
                seq_len,
                train_bincounts,
                data_name=data_name,
                **model_config
            )
        elif model_name == "routercl4kt":
            model_config = config.routercl4kt_config
            # Validate head configuration
            validate_head_config(
                model_config.num_attn_heads,
                model_config.num_shared_heads,
                model_config.num_selected_heads
            )
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
            model = RouterCL4KT(
                device,
                num_skills,
                num_questions,
                seq_len,
                train_bincounts,
                **model_config
            )

        with open(os.path.join(dir_name, "configs.json"), 'w') as f:
            json.dump(model_config, f)
            json.dump(train_config, f)

        print(train_config)
        print(model_config)

        if model_name == "cl4kt" or model_name == "routercl4kt":
            train_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        train_dataset,
                        seq_len,
                        mask_prob,
                        crop_prob,
                        permute_prob,
                        replace_prob,
                        negative_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    MKMDatasetWrapper(
                        diff_order, valid_dataset, seq_len, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    MKMDatasetWrapper(
                        diff_order, test_dataset, seq_len, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "dis" in model_name:  # diskt
            train_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        train_dataset,
                        seq_len,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        valid_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        test_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        else:
            train_loader = accelerator.prepare(
                DataLoader(train_dataset, batch_size=batch_size)
            )

            valid_loader = accelerator.prepare(
                DataLoader(valid_dataset, batch_size=eval_batch_size)
            )

            test_loader = accelerator.prepare(
                DataLoader(test_dataset, batch_size=eval_batch_size)
            )

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate, weight_decay=train_config.l2)

        model, opt = accelerator.prepare(model, opt)

        t1 = model_train(
            dir_name,
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
        ) #t1 = [test_auc, test_acc, test_rmse]

        test_result.append(t1) # fold, 9

    print_args = dict()
    metric_type = ['d', 'b', 'w']
    metric = ['auc', 'acc', 'rmse']

    # Original metrics calculation for training mode
    for index in range(len(test_result[0])):
        fold_total = []
        for fold in range(len(test_result)):
            fold_total.append(test_result[fold][index])
        print_args[f'{metric[index%3]}_{metric_type[index//3]}'] = np.mean(fold_total)

    # Create results table data
    data = [[model_name, data_name]]
    columns = ["Model", "Dataset"]

    # For training mode, log all metrics
    for index in range(len(test_result[0])):
        fold_total = []
        for fold in range(len(test_result)):
            fold_total.append(test_result[fold][index])
        metric_name = f'{metric[index%3]}_{metric_type[index//3]}'
        mean_value = np.mean(fold_total)
        std_value = np.std(fold_total)

        columns.append(f"{metric_name}")
        data[0].append(f"{mean_value:.4f}±{std_value:.4f}")

    if config.use_wandb:

        # Create wandb table
        table = wandb.Table(data=data, columns=columns)
        wandb.log({"Results Table": table})

        # 同时也记录单独的指标
        print_args['Model'] = model_name
        print_args['Dataset'] = data_name
        print_args.update(train_config)
        print_args.update(model_config)
        wandb.log(print_args)
    
    # print auc_d, acc_d, rmse_d
    print(f"Overall Performance \n AUC: {print_args['auc_d']:.4f}, ACC: {print_args['acc_d']:.4f}, RMSE: {print_args['rmse_d']:.4f}")


if __name__ == "__main__":
    # First load the config file to get defaults
    base_cfg_file = PathManager.open("configs/opt.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="cl4kt",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt, sakt, simplekt, routerkt, routerakt, routersakt, routercl4kt]. \
            The default model is cl4kt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="algebra05",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument(
        "--reg_l",
        type=float,
        default=0.1,
        help="regularization parameter learning loss",
    )
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float, default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--inter_lambda", type=float, default=1, help="loss lambda ratio for regularization"
    )
    parser.add_argument(
        "--ques_lambda", type=float, default=1, help="loss lambda ratio for regularization"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument(
        "--only_rp", type=int, default=0, help="train with only rp model"
    )
    parser.add_argument(
        "--choose_cl", type=str, default="both", help="choose between q_cl and s_cl"
    )
    parser.add_argument(
        "--describe", type=str, default="default", help="description of the training"
    )
    parser.add_argument(
        "--diff_order", type=str, default="random", help="random/des/asc/chunk"
    )
    parser.add_argument(
        "--use_wandb", type=int, default=0
    )
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")

    parser.add_argument("--total_cnt_init", type=int, default=0, help="total_cnt_init")
    parser.add_argument("--diff_unk", type=float, default=0.5, help="diff_unk")

    parser.add_argument("--gpu_num", type=int, default=0, help="gpu number")
    parser.add_argument("--server_num", type=str, default="0", help="server number")

    parser.add_argument("--diff_as_loss_weight", action="store_true", default=False, help="diff_as_loss_weight")
    parser.add_argument("--valid_balanced", action="store_true", default=False, help="valid_balanced")
    parser.add_argument("--exponential", action="store_true", default=True, help="exponential function for forgetting behavior")
    parser.add_argument("--uniform", action="store_true", default=False, help="uniform or quantiles for difficulty")
    parser.add_argument("--seed",  type=int, default=12405, help="seed")

    parser.add_argument("--de_type", type=str, default="none_0", help="difficulty encoding")
    parser.add_argument("--choose_enc", type=str, default="f", help="choose encoder")

    parser.add_argument("--seq_len",  type=int, default=100, help="max sequence length")
    parser.add_argument(
        "--balance_loss_weight", 
        type=float, 
        default=0.001, 
        help="weight for the balance loss in RouterAlibiKT"
    )
    parser.add_argument(
        "--num_shared_heads", 
        type=int, 
        default=2, 
        help="number of shared heads"
    )
    parser.add_argument(
        "--num_attn_heads", 
        type=int, 
        default=8, 
        help="number of attention heads"
    )
    parser.add_argument(
        "--num_selected_heads",
        type=int,
        default=2,
        help="number of attention heads to select in MoH attention"
    )
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="query_norm",
        choices=["dynamic", "query_norm"],
        help="Mode for attention head routing: dynamic (learned) or query_norm"
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64,
        help="Size of the embedding dimension"
    )
    parser.add_argument(
        "--state_d",
        type=int,
        default=64,
        help="Size of the embedding dimension"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=300,
        help="Number of epochs to train the model"
    )

    args = parser.parse_args()

    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.use_wandb = args.use_wandb
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    cfg.train_config.describe = args.describe
    cfg.train_config.gpu_num = args.gpu_num
    cfg.train_config.server_num = args.server_num
    cfg.train_config.diff_as_loss_weight = args.diff_as_loss_weight
    cfg.train_config.valid_balanced = args.valid_balanced
    cfg.train_config.uniform = args.uniform
    cfg.train_config.seed = args.seed
    cfg.train_config.seq_len = args.seq_len
    cfg.train_config.num_epochs = args.num_epochs
    cfg.train_config.l2 = args.l2

    cfg.total_cnt_init = args.total_cnt_init
    cfg.diff_unk = args.diff_unk

    if args.model_name == "cl4kt":
        cfg.cl4kt_config = cfg.cl4kt_config
        cfg.cl4kt_config.only_rp = args.only_rp
        cfg.cl4kt_config.choose_cl = args.choose_cl
    elif args.model_name == "akt":
        cfg.akt_config = cfg.akt_config
    elif args.model_name == "routerakt":
        if args.data_name in cfg.routerakt_config:
            cfg.routerakt_config = cfg.routerakt_config[args.data_name]
        # Update config with command line args if provided
        if args.balance_loss_weight != 0.001:  # default value
            cfg.routerakt_config.balance_loss_weight = args.balance_loss_weight
        if args.num_shared_heads != 2:  # default value
            cfg.routerakt_config.num_shared_heads = args.num_shared_heads
        if args.num_attn_heads != 8:  # default value
            cfg.routerakt_config.num_attn_heads = args.num_attn_heads
        if args.num_selected_heads != 2:  # default value
            cfg.routerakt_config.num_selected_heads = args.num_selected_heads
        if args.routing_mode != "query_norm":  # default value
            cfg.routerakt_config.routing_mode = args.routing_mode
        if args.de_type != "none_0":
            cfg.routerakt_config.de_type = args.de_type
        if args.choose_enc != "f":
            cfg.routerakt_config.choose_enc = args.choose_enc
    elif args.model_name == "routersimplekt":
        cfg.routersimplekt_config = cfg.routersimplekt_config if hasattr(cfg, 'routersimplekt_config') else cfg.simplekt_config
        # Update config with command line args if provided
        if args.balance_loss_weight != 0.001:
            cfg.routersimplekt_config.balance_loss_weight = args.balance_loss_weight
        if args.num_shared_heads != 2:
            cfg.routersimplekt_config.num_shared_heads = args.num_shared_heads
        if args.num_attn_heads != 8:
            cfg.routersimplekt_config.num_attn_heads = args.num_attn_heads
        if args.num_selected_heads != 2:
            cfg.routersimplekt_config.num_selected_heads = args.num_selected_heads
        if args.routing_mode != "query_norm":
            cfg.routersimplekt_config.routing_mode = args.routing_mode
        if args.de_type != "none_0":
            cfg.routersimplekt_config.de_type = args.de_type
        if args.choose_enc != "f":
            cfg.routersimplekt_config.choose_enc = args.choose_enc
    elif args.model_name == "routercl4kt":
        cfg.routercl4kt_config = cfg.routercl4kt_config if hasattr(cfg, 'routercl4kt_config') else cfg.cl4kt_config
        # Update config with command line args if provided
        if args.balance_loss_weight != 0.001:
            cfg.routercl4kt_config.balance_loss_weight = args.balance_loss_weight
        if args.num_shared_heads != 2:
            cfg.routercl4kt_config.num_shared_heads = args.num_shared_heads
        if args.num_attn_heads != 8:
            cfg.routercl4kt_config.num_attn_heads = args.num_attn_heads
        if args.num_selected_heads != 2:
            cfg.routercl4kt_config.num_selected_heads = args.num_selected_heads
        if args.routing_mode != "query_norm":
            cfg.routercl4kt_config.routing_mode = args.routing_mode
        # Add only_rp and choose_cl to the config before freezing
        cfg.routercl4kt_config.only_rp = args.only_rp
        cfg.routercl4kt_config.choose_cl = args.choose_cl
    elif args.model_name == "simplekt":
        cfg.simplekt_config = cfg.simplekt_config
    elif args.model_name == "sakt":
        cfg.sakt_config = cfg.sakt_config

    if args.model_name in ["akt", "cl4kt", "sakt", "simplekt"]:
        cfg[f"{args.model_name}_config"].de_type = args.de_type
        cfg[f"{args.model_name}_config"].choose_enc = args.choose_enc
    # Also explicitly set de_type and choose_enc for router models
    elif args.model_name in ["routerakt", "routercl4kt", "routersimplekt"]:
        if not hasattr(cfg[f"{args.model_name.replace('router', '')}_config"], 'de_type'):
            cfg[f"{args.model_name.replace('router', '')}_config"].de_type = args.de_type
        if not hasattr(cfg[f"{args.model_name.replace('router', '')}_config"], 'choose_enc'):
            cfg[f"{args.model_name.replace('router', '')}_config"].choose_enc = args.choose_enc

    cfg.seed = args.seed
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.describe = args.describe
    cfg.total_cnt_init = args.total_cnt_init
    cfg.diff_unk = args.diff_unk
    cfg.use_wandb = args.use_wandb

    cfg.freeze()
    main(cfg)
