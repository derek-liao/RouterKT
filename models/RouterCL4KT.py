import math
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Embedding,
    Linear,
    ReLU,
    Dropout,
    ModuleList,
    Softplus,
    Sequential,
    Sigmoid,
    BCEWithLogitsLoss,
    LayerNorm,
)
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from .modules import transformer_FFN, Similarity, MoHAttention
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
from .rpe import SinusoidalPositionalEmbeddings 

class RouterCL4KT(Module):
    def __init__(self, device, num_skills, num_questions, seq_len, bincounts, **kwargs):
        super(RouterCL4KT, self).__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.bincounts = bincounts
        self.args = kwargs
        self.hidden_size = self.args["hidden_size"]
        self.num_blocks = self.args["num_blocks"]
        self.num_attn_heads = self.args["num_attn_heads"]
        self.num_shared_heads = self.args.get("num_shared_heads", 2)  # Default to 2 shared heads
        self.num_selected_heads = self.args.get("num_selected_heads", 4)  # Default to 4 selected heads
        self.routing_mode = self.args.get("routing_mode", "dynamic")  # Add new parameter with default
        self.kq_same = self.args["kq_same"]
        self.final_fc_dim = self.args["final_fc_dim"]
        self.d_ff = self.args["d_ff"]
        self.dropout = self.args["dropout"]
        self.reg_cl = self.args["reg_cl"]
        self.negative_prob = self.args["negative_prob"]
        self.hard_negative_weight = self.args["hard_negative_weight"]
        self.only_rp = self.args["only_rp"]
        self.choose_cl = self.args["choose_cl"]
        self.balance_loss_weight = self.args.get("balance_loss_weight", 0.01)  # Default to 0.01

        # Validate head configurations
        if self.num_attn_heads < self.num_shared_heads:
            raise ValueError(f"Total attention heads ({self.num_attn_heads}) must be greater than shared heads ({self.num_shared_heads})")
        if self.num_selected_heads > (self.num_attn_heads - self.num_shared_heads):
            raise ValueError(f"Selected heads ({self.num_selected_heads}) cannot exceed number of dynamic heads ({self.num_attn_heads - self.num_shared_heads})")
            
        self.position_emb = Embedding(seq_len + 1, self.hidden_size, padding_idx=0)
        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )
        self.sim = Similarity(temp=self.args["temp"])

        # Create transformer blocks with MoH attention
        self.question_encoder = ModuleList(
            [
                RouterCL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    n_heads=self.num_attn_heads,
                    n_shared_heads=self.num_shared_heads,
                    n_selected_heads=self.num_selected_heads,
                    seq_len=seq_len,
                    bincounts=self.bincounts,
                    routing_mode=self.routing_mode,
                    kq_same=self.kq_same
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                RouterCL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    n_heads=self.num_attn_heads,
                    n_shared_heads=self.num_shared_heads,
                    n_selected_heads=self.num_selected_heads,
                    seq_len=seq_len,
                    bincounts=self.bincounts,
                    routing_mode=self.routing_mode,
                    kq_same=self.kq_same
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                RouterCL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    n_heads=self.num_attn_heads,
                    n_shared_heads=self.num_shared_heads,
                    n_selected_heads=self.num_selected_heads,
                    seq_len=seq_len,
                    bincounts=self.bincounts,
                    routing_mode=self.routing_mode,
                    kq_same=self.kq_same
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )

        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, batch):
        if self.training:
            q_i, q_j, q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r_i, r_j, r, neg_r = batch[
                "responses"
            ]  # augmented r_i, augmented r_j and original r
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"]
            diff_i, diff_j, diff = batch["sdiff"]
            diff_i, diff_j, diff = (diff_i*(r_i>-1).int()).long(), (diff_j*(r_j>-1).int()).long(), (diff*(r>-1).int()).long()    
            pos = batch["position"]
            
            if not self.only_rp:
                ques_i_embed = self.question_embed(q_i)
                ques_j_embed = self.question_embed(q_j)
                inter_i_embed = self.get_interaction_embed(q_i, r_i)
                inter_j_embed = self.get_interaction_embed(q_j, r_j)
                inter_k_embed = self.get_interaction_embed(q, neg_r)
                    
                q_i_enc, q_j_enc = None, None
                i_i_enc, i_j_enc, i_k_enc = None, None, None   

                # mask=2 means bidirectional attention of BERT
                ques_i_score, ques_j_score = ques_i_embed, ques_j_embed
                inter_i_score, inter_j_score = inter_i_embed, inter_j_embed

                # BERT
                if self.choose_cl in ["q_cl", "both"]:
                    for block in self.question_encoder:
                        ques_i_score, _ = block(
                            mask=2,
                            query=ques_i_score,
                            key=ques_i_score,
                            values=ques_i_score,
                            diff=q_i_enc,
                            response=r_i,
                        )
                        ques_j_score, _ = block(
                            mask=2,
                            query=ques_j_score,
                            key=ques_j_score,
                            values=ques_j_score,
                            diff=q_j_enc,
                            response=r_j,
                        )
                if self.choose_cl in ["s_cl", "both"]:
                    for block in self.interaction_encoder:
                        inter_i_score, _ = block(
                            mask=2,
                            query=inter_i_score,
                            key=inter_i_score,
                            values=inter_i_score,
                            diff=i_i_enc,
                            response=r_i,
                        )
                        inter_j_score, _ = block(
                            mask=2,
                            query=inter_j_score,
                            key=inter_j_score,
                            values=inter_j_score,
                            diff=i_j_enc,
                            response=r_j,
                        )
                        if self.negative_prob > 0:
                            inter_k_score, _ = block(
                                mask=2,
                                query=inter_k_embed,
                                key=inter_k_embed,
                                values=inter_k_embed,
                                diff=i_k_enc,
                                response=neg_r,
                            )
                if self.choose_cl in ["q_cl", "both"]:
                    pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_i.sum(-1).unsqueeze(-1)
                    pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_j.sum(-1).unsqueeze(-1)

                    ques_cos_sim = self.sim(
                        pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0)
                    )
                    # Hard negative should be added

                    ques_labels = torch.arange(ques_cos_sim.size(0)).long().to(q_i.device)
                    question_cl_loss = self.cl_loss_fn(ques_cos_sim, ques_labels)
                    # question_cl_loss = torch.mean(question_cl_loss)
                else: 
                    question_cl_loss = 0
                if self.choose_cl in ["s_cl", "both"]:
                    pooled_inter_i_score = (inter_i_score * attention_mask_i.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_i.sum(-1).unsqueeze(-1)
                    pooled_inter_j_score = (inter_j_score * attention_mask_j.unsqueeze(-1)).sum(
                        1
                    ) / attention_mask_j.sum(-1).unsqueeze(-1)

                    inter_cos_sim = self.sim(
                        pooled_inter_i_score.unsqueeze(1), pooled_inter_j_score.unsqueeze(0)
                    )

                    if self.negative_prob > 0:
                        pooled_inter_k_score = (
                            inter_k_score * attention_mask.unsqueeze(-1)
                        ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                        neg_inter_cos_sim = self.sim(
                            pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0)
                        )
                        inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)

                    inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(q_i.device)

                    if self.negative_prob > 0:
                        weights = torch.tensor(
                            [
                                [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                                + [0.0] * i
                                + [self.hard_negative_weight]
                                + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                                for i in range(neg_inter_cos_sim.size(-1))
                            ]
                        ).to(q_i.device)
                        inter_cos_sim = inter_cos_sim + weights

                    interaction_cl_loss = self.cl_loss_fn(inter_cos_sim, inter_labels)
                else: 
                    interaction_cl_loss = 0 
            else: 
                question_cl_loss, interaction_cl_loss = 0, 0
        else:
            q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r = batch["responses"]  # augmented r_i, augmented r_j and original r

            attention_mask = batch["attention_mask"]
            diff = batch["sdiff"]
            diff = (diff*(r>-1).int()).long()    
            pos = batch["position"]
            question_cl_loss, interaction_cl_loss = 0, 0
            
        q_embed = self.question_embed(q)
        i_embed = self.get_interaction_embed(q, r)
        f_embed = None 
            
        q_enc = None
        i_enc = None 
        f_enc = None 
                
        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, diff=q_enc, response=r, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, diff=i_enc, response=r, apply_pos=True)

        for idx, block in enumerate(self.knoweldge_retriever):
            if f_embed is not None:
                x = x+f_embed
                y = y+f_embed
            x, attn = block(mask=0, query=x, key=x, values=y, diff=f_enc, response=r, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()
        total_cl_loss = question_cl_loss + interaction_cl_loss

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "cl_loss": total_cl_loss,
                "attn": attn,
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "attn": attn,
                "x": x,
            }

        return out_dict

    def get_balance_loss(self):
        """Get total balance loss from all transformer layers."""
        balance_loss = 0
        for block in self.question_encoder:
            balance_loss += block.attn.get_balance_loss()
        for block in self.interaction_encoder:
            balance_loss += block.attn.get_balance_loss()
        for block in self.knoweldge_retriever:
            balance_loss += block.attn.get_balance_loss()
        return balance_loss

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        if not self.only_rp:
            cl_loss = torch.mean(out_dict["cl_loss"])  # torch.mean() for multi-gpu FIXME
        else:
            cl_loss = 0
        mask = true > -1

        # Get balance loss using the new method
        balance_loss = self.get_balance_loss()

        # Total loss combines prediction loss, contrastive loss, and balance loss
        loss = self.loss_fn(pred[mask], true[mask]) + self.reg_cl * cl_loss + self.balance_loss_weight * balance_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses):
        masked_responses = responses * (responses > -1).long()
        interactions = skills + self.num_skills * masked_responses
        output = self.interaction_embed(interactions)
        return output
    
class RouterCL4KTTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, dropout, n_heads, 
                 n_shared_heads, n_selected_heads, seq_len, bincounts=None, routing_mode="dynamic", kq_same=True):
        super().__init__()
        
        # Multi-head attention with dynamic routing
        self.attn = MoHAttention(
            d_model=d_model,
            d_feature=d_feature,
            n_heads=n_heads,
            n_shared_heads=n_shared_heads,
            n_selected_heads=n_selected_heads,
            dropout=dropout,
            kq_same=kq_same,
            seq_len=seq_len,
            routing_mode=routing_mode
        )
        
        # Layer norm and dropout
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        
        # Feed forward network
        self.ffn = transformer_FFN(d_model, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, mask, query, key, values, diff=None, response=None, apply_pos=True):
        # Create proper attention mask based on the mask parameter
        seqlen = query.size(1)
        if mask == 0:  # can only see past values
            nopeek_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()
        else:  # mask == 1 or 2, can see current and past values
            nopeek_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool()
            
        src_mask = (~nopeek_mask).to(query.device)
        
        # Apply MoH attention with proper masking
        attn_output = self.attn(query, key, values, src_mask)
        x = self.layer_norm1(query + self.dropout1(attn_output))
        
        if apply_pos:
            # Feed forward
            ffn_output = self.ffn(x)
            x = self.layer_norm2(x + self.dropout2(ffn_output))
        
        return x, self.attn.get_balance_loss() 