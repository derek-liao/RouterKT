import pandas as pd
import numpy as np
import torch
import os
import glob
import csv
import json
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time
from utils.utils import calculate_balance_loss

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def model_train(
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
    early_stop=True,
):
    train_losses = []
    avg_train_losses = []
    best_valid_auc = 0

    # Add balance loss tracking
    balance_losses = []
    avg_balance_losses = []

    logs_df = pd.DataFrame()
    num_epochs = config["train_config"]["num_epochs"]
    model_name = config["model_name"]
    data_name = config["data_name"]
    train_config = config["train_config"]
    valid_balanced = train_config["valid_balanced"]

    # Lists to store metrics for plotting
    epoch_train_losses = []
    epoch_valid_aucs = []
    epoch_balance_losses = []  # New list for balance losses

    token_cnts = 0
    label_sums = 0
    best_epoch = 0
    for i in range(1, num_epochs + 1):
        model.train()
        opt.zero_grad() 
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            out_dict = model(batch)
            if torch.cuda.device_count() > 1 and hasattr(model, 'module'):
                loss, token_cnt, label_sum = model.module.loss(batch, out_dict)
            else:
                loss, token_cnt, label_sum = model.loss(batch, out_dict)
            
            # Calculate balance loss
            balance_loss = calculate_balance_loss(model, model_name, batch, out_dict)
            
            accelerator.backward(loss)
            
            token_cnts += token_cnt
            label_sums += label_sum
            balance_losses.append(balance_loss.item())

            # Apply gradient clipping if configured
            if train_config["max_grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config["max_grad_norm"]
                )
                
            # Step the optimizer on every batch
            opt.step()
            opt.zero_grad()

            train_losses.append(loss.item())

        print("token_cnts", token_cnts, "label_sums", label_sums)

        total_preds, total_trues = [], []

        with torch.no_grad():
            for batch in valid_loader:
                model.eval()

                out_dict = model(batch)
                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                pred = pred[mask]
                true = true[mask]

                total_preds.append(pred)
                total_trues.append(true)

            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        train_loss = np.average(train_losses)
        balance_loss = np.average(balance_losses)  # Calculate average balance loss
        avg_train_losses.append(train_loss)
        avg_balance_losses.append(balance_loss)  # Track average balance loss
        valid_auc_balanced = 0
        valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        # Store metrics for plotting
        epoch_train_losses.append(train_loss)
        epoch_valid_aucs.append(valid_auc)
        epoch_balance_losses.append(balance_loss)  # Store balance loss for plotting

        if valid_balanced:
            early_stop_valid = valid_auc_balanced
        else:
            early_stop_valid = valid_auc

        path = os.path.join("saved_model", model_name, data_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        if early_stop_valid > best_valid_auc:
            path = os.path.join(
                dir_name, f"{fold}_params_*"
            )
            for _path in glob.glob(path):
                os.remove(_path)
            best_valid_auc = early_stop_valid
            best_epoch = i
            torch.save(
                {"epoch": i, "model_state_dict": model.state_dict()},
                os.path.join(dir_name, f"{fold}_params_best.pt")
                )
        
        if early_stop and i - best_epoch > 10:
            break

        # clear lists to track next epochs
        train_losses = []
        balance_losses = []  # Clear balance losses for next epoch

        total_preds, total_trues = [], []

        print(f"Fold {fold}:\t Epoch {i}\tTRAIN LOSS: {train_loss:.4f}\tVALID AUC: {valid_auc:.4f}\tVALID AUC(Balanced): {valid_auc_balanced:.4f}\tBALANCE LOSS: {balance_loss:.4f}")
        
    checkpoint = torch.load(os.path.join(dir_name, f"{fold}_params_best.pt"))

    model.load_state_dict(checkpoint["model_state_dict"])

    total_preds, total_trues = [], []

    # evaluation on test dataset
    with torch.no_grad():
        for batch in test_loader:
            model.eval()

            out_dict = model(batch)

            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            mask = true > -1
            pred = pred[mask]
            true = true[mask]
            total_preds.append(pred)
            total_trues.append(true)

        total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
        total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

    auc, acc, rmse, auc_weighted, acc_weighted, rmse_weighted = calculate_metrics(total_trues, total_preds)

    auc_balanced = 0
    acc_balanced = 0
    rmse_balanced = 0
    
    # print(f"[ORIGINAL] Best Model\tTEST AUC: {auc:.4f}\tTEST ACC: {acc:.4f}\tTEST RMSE: {rmse:.4f}")

    # Return the test metrics
    return [auc, acc, rmse, auc_balanced, acc_balanced, rmse_balanced, auc_weighted, acc_weighted, rmse_weighted]

def calculate_metrics(total_trues, total_preds):
    """Calculate AUC, ACC, RMSE, and weighted versions."""
    auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
    acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
    rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))
    
    sw = np.where(total_trues == 1, 1-sum(total_trues)/len(total_trues), sum(total_trues)/len(total_trues))
    auc_weighted = roc_auc_score(y_true=total_trues, y_score=total_preds, sample_weight=sw)
    acc_weighted = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5, sample_weight=sw)
    rmse_weighted = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds, sample_weight=sw))
    
    return auc, acc, rmse, auc_weighted, acc_weighted, rmse_weighted

