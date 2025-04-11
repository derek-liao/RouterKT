import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch import FloatTensor, LongTensor
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import wandb
import time
import seaborn as sns
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# the following collate functions are based on the code:
# https://github.com/hcnoh/knowledge-tracing-collection-pytorch
def collate_question_response_fn(batches, pad_val=-1):
    questions = []
    responses = []
    targets = []
    deltas = []

    for batch in batches:
        questions.append(LongTensor(batch["questions"][:-1]))
        responses.append(LongTensor(batch["responses"][:-1]))
        targets.append(FloatTensor(batch["responses"][1:]))
        deltas.append(LongTensor(batch["questions"][1:]))

    """
    pad_sequence를 통해 list of LongTensor가 [B x L (=200)] 의 Tensor로 변환됨
    """
    questions = pad_sequence(questions, batch_first=True, padding_value=pad_val)
    responses = pad_sequence(responses, batch_first=True, padding_value=pad_val)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_val)
    deltas = pad_sequence(deltas, batch_first=True, padding_value=pad_val)

    masks = (questions != pad_val) * (deltas != pad_val)

    questions, responses, targets, deltas = (
        questions * masks,
        responses * masks,
        targets * masks,
        deltas * masks,
    )


    return questions, responses, targets, deltas, masks


def collate_question_skill_response_fn(batches, pad_val=-1):
    questions = []
    skills = []
    responses = []
    targets = []
    delta_questions = []
    delta_skills = []

    for batch in batches:
        questions.append(LongTensor(batch["questions"][:-1]))
        skills.append(LongTensor(batch["skills"][:-1]))
        responses.append(LongTensor(batch["responses"][:-1]))
        targets.append(FloatTensor(batch["responses"][1:]))
        delta_questions.append(LongTensor(batch["questions"][1:]))
        delta_skills.append(LongTensor(batch["skills"][1:]))

    questions = pad_sequence(questions, batch_first=True, padding_value=pad_val)
    skills = pad_sequence(skills, batch_first=True, padding_value=pad_val)
    responses = pad_sequence(responses, batch_first=True, padding_value=pad_val)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_val)
    delta_questions = pad_sequence(
        delta_questions, batch_first=True, padding_value=pad_val
    )
    delta_skills = pad_sequence(delta_skills, batch_first=True, padding_value=pad_val)

    masks = (questions != pad_val) * (delta_questions != pad_val)

    questions, skills, responses, targets, delta_questions, delta_skills = (
        questions * masks,
        skills * masks,
        responses * masks,
        targets * masks,
        delta_questions * masks,
        delta_skills * masks,
    )


    return questions, skills, responses, targets, delta_questions, delta_skills, masks


def collate_fn(batches):
    questions = []
    skills = []
    responses = []
    lengths = []

    for batch in batches:
        questions.append(LongTensor(batch["questions"]))
        skills.append(LongTensor(batch["skills"]))
        responses.append(LongTensor(batch["responses"]))

    questions = pad_sequence(questions, batch_first=True, padding_value=0)
    skills = pad_sequence(skills, batch_first=True, padding_value=0)
    responses = pad_sequence(responses, batch_first=True, padding_value=-1)

    feed_dict = {"questions": questions, "skills": skills, "responses": responses}
    return feed_dict

def calculate_balance_loss(model, model_name, batch, out_dict):
    """
    Calculate the balance loss based on the model architecture.
    """
    if torch.cuda.device_count() > 1 and hasattr(model, 'module'):
        if model_name.lower() == "routerakt":
            balance_loss = model.module.model.get_balance_loss() if hasattr(model.module.model, 'get_balance_loss') else torch.tensor(0.0).to(out_dict["pred"].device)
        elif model_name.lower() in ["routercl4kt", "routersimplekt"]:
            balance_loss = model.module.get_balance_loss() if hasattr(model.module, 'get_balance_loss') else torch.tensor(0.0).to(out_dict["pred"].device)
        else:
            balance_loss = torch.tensor(0.0).to(out_dict["pred"].device)
    else:
        if model_name.lower() == "routerakt":
            balance_loss = model.model.get_balance_loss() if hasattr(model.model, 'get_balance_loss') else torch.tensor(0.0).to(out_dict["pred"].device)
        elif model_name.lower() in ["routercl4kt", "routersimplekt"]:
            balance_loss = model.get_balance_loss() if hasattr(model, 'get_balance_loss') else torch.tensor(0.0).to(out_dict["pred"].device)
        else:
            balance_loss = torch.tensor(0.0).to(out_dict["pred"].device)
    return balance_loss


def plot_training_curves(config, fold, epoch_train_losses, epoch_valid_aucs, epoch_balance_losses, best_epoch, model_name, best_valid_auc):
    """
    Plot training curves and log them to wandb.
    """
    if config.get("use_wandb", False):
        # Set academic style for plots
        # plt.style.use('seaborn-v0_8-whitegrid')
        
        # Define academic color palette (colorblind-friendly)
        colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC']
        
        # Create figure with appropriate size for academic papers
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
        
        epochs = range(1, len(epoch_train_losses) + 1)
        
        # Create twin axes for different metrics
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the right twin axis to make room for the third axis
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot training loss on left axis
        line1 = ax1.plot(epochs, epoch_train_losses, 'o-', color=colors[0], 
                       linewidth=2, markersize=4, label='Train Loss')
        ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Train Loss', fontsize=10, fontweight='bold', color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])
        
        # Plot validation AUC on first right axis
        line2 = ax2.plot(epochs, epoch_valid_aucs, 's-', color=colors[1], 
                       linewidth=2, markersize=4, label='Valid AUC')
        ax2.set_ylabel('Valid AUC', fontsize=10, fontweight='bold', color=colors[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])
        
        # Plot balance loss on second right axis
        line3 = ax3.plot(epochs, epoch_balance_losses, 'd-', color=colors[2],
                       linewidth=2, markersize=4, label='Balance Loss')
        ax3.set_ylabel('Balance Loss', fontsize=10, fontweight='bold', color=colors[2])
        ax3.tick_params(axis='y', labelcolor=colors[2])
        
        # Set title
        plt.title('Training Dynamics', fontsize=12, fontweight='bold', pad=10)
        
        # Add fold information as text annotation
        ax1.text(0.02, 0.02, f'Fold: {fold}', transform=ax1.transAxes, 
                fontsize=8, fontweight='bold', color='dimgrey')
        
        # Add model name as text annotation
        ax1.text(0.98, 0.02, f'Model: {model_name}', transform=ax1.transAxes, 
                fontsize=8, fontweight='bold', color='dimgrey', ha='right')
        
        # Highlight best epoch with vertical line
        ax1.axvline(x=best_epoch, color='grey', linestyle='--', alpha=0.7, linewidth=1)
        ax1.text(best_epoch + 0.1, min(epoch_train_losses), f'Best Epoch: {best_epoch}', 
                fontsize=8, color='dimgrey')
        
        # Combine legends from all axes
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', fontsize=9, 
                  frameon=True, edgecolor='black', fancybox=False)
        
        # Customize grid
        ax1.grid(axis='both', linestyle='--', alpha=0.3, color='grey')
        
        # Customize ticks
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        
        # Set x-ticks to show only integer values
        ax1.set_xticks(np.arange(1, len(epoch_train_losses) + 1, max(1, len(epoch_train_losses) // 10)))
        
        # Add a light box around the plot
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('black')
        
        plt.tight_layout()
        
        # Log the plot to wandb
        wandb.log({f"fold_{fold}/training_curves": wandb.Image(fig)})
        plt.close(fig)
        
        # Also log a summary of the best performance
        wandb.log({
            f"fold_{fold}/best_valid_auc": best_valid_auc,
            f"fold_{fold}/best_epoch": best_epoch,
            f"fold_{fold}/final_balance_loss": epoch_balance_losses[-1]
        })

