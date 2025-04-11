import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss, ModuleList, Parameter
from torch.nn.functional import binary_cross_entropy
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MoHAttention, CosinePositionalEmbedding
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn.functional as F
import numpy as np
from .rpe import RotaryPositionalEmbeddings

class RouterSimpleKT(Module):
    def __init__(self, device, num_skills, num_questions, seq_len,
                 embedding_size, num_blocks, dropout, d_ff=256, 
                 num_attn_heads=8, num_shared_heads=2, num_selected_heads=4,
                 num_alibi_heads=0,
                 separate_qa=False, l2=1e-5,
                 final_fc_dim=512, final_fc_dim2=256, balance_loss_weight=0.01,
                 routing_mode="dynamic", kq_same=True, **kwargs):
        super().__init__()
        self.model_name = "routersimplekt"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.dropout = dropout
        self.l2 = l2
        self.separate_qa = separate_qa
        self.balance_loss_weight = balance_loss_weight
        self.num_alibi_heads = num_alibi_heads
        self.routing_mode = routing_mode
        self.kq_same = kq_same
        
        # Question difficulty parameters if using question IDs
        if self.num_questions > 0:
            self.difficult_param = Embedding(self.num_questions + 1, 1)  # μ_q
            self.q_embed_diff = Embedding(self.num_skills + 1, embedding_size)  # d_c
            self.qa_embed_diff = Embedding(2 * self.num_skills + 1, embedding_size)  # f_(c,r)
        
        # Basic embeddings
        self.q_embed = Embedding(num_skills, embedding_size)  # c_c
        if self.separate_qa:
            self.qa_embed = Embedding(2 * num_skills + 1, embedding_size)
        else:
            self.qa_embed = Embedding(2, embedding_size)  # g_r
            
        # Main architecture with MoH attention
        self.model = Architecture(
            n_question=num_skills,
            n_blocks=num_blocks,
            n_heads=num_attn_heads,
            n_shared_heads=num_shared_heads,
            n_selected_heads=num_selected_heads,
            dropout=dropout,
            d_model=embedding_size,
            d_feature=embedding_size // num_attn_heads,
            d_ff=d_ff,
            seq_len=seq_len,
            routing_mode=routing_mode,
            kq_same=kq_same
        )
        
        # Output layers
        self.out = nn.Sequential(
            Linear(embedding_size * 2, final_fc_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            Dropout(dropout),
            Linear(final_fc_dim2, 1)
        )

        # self.out = Linear(embedding_size * 2, 1)
        
        self.loss_fn = BCELoss(reduction="mean")
        self.reset()
        
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions + 1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, feed_dict):
        # Get current and previous timestep information
        q = feed_dict["skills"][:, :-1]  # Previous skills
        qnext = feed_dict["skills"][:, 1:]  # Current skills to predict
        r = feed_dict["responses"][:, :-1]  # Previous responses
        attention_mask = feed_dict["attention_mask"][:, :-1]  # Adjust mask accordingly
        masked_r = r * (r > -1).long()
        pid_data = feed_dict["questions"][:, :-1] if "questions" in feed_dict else None
        diff = feed_dict["sdiff"] if "sdiff" in feed_dict else None
        if diff is not None:
            diff = (diff*(feed_dict["responses"]>-1).int()).long()
        
        # Get base embeddings for previous timesteps
        q_embed_data = self.q_embed(q)  # c_c
        if self.separate_qa:
            qa_data = q + self.num_skills * masked_r
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = q_embed_data + self.qa_embed(masked_r)  # c_c + g_r
            
        # Add question difficulty if using question IDs
        if self.num_questions > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(q)  # d_c
            pid_embed_data = self.difficult_param(pid_data)  # μ_q
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # c_c + μ_q * d_c
            
            qa_embed_diff_data = self.qa_embed_diff(q + self.num_skills * masked_r)  # f_(c,r)
            qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            
        # Get embeddings for current skills to predict
        qnext_embed_data = self.q_embed(qnext)
            
        # Pass through transformer with MoH attention
        d_output, balance_loss = self.model(q_embed_data, qa_embed_data, diff, r)
        
        # Final prediction
        concat_q = torch.cat([d_output, qnext_embed_data], dim=-1)
        output = torch.sigmoid(self.out(concat_q)).squeeze(-1)
        
        out_dict = {
            "pred": output,
            "true": feed_dict["responses"][:, 1:].float(),
            "balance_loss": balance_loss
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        
        # Main prediction loss
        pred_loss = self.loss_fn(pred[mask], true[mask])
        
        # Add balance loss from MoH layers
        balance_loss = out_dict["balance_loss"]
        
        # Total loss with balance loss weighted by beta=0.01
        total_loss = pred_loss + self.l2 * torch.norm(self.difficult_param.weight) + self.balance_loss_weight * balance_loss
        
        return total_loss, len(pred[mask]), true[mask].sum().item()
    
    def get_balance_loss(self):
        """Get the total balance loss from all transformer blocks."""
        total_balance_loss = 0
        for block in self.model.blocks:
            total_balance_loss += block.attn.get_balance_loss()
        return total_balance_loss

class Architecture(Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, n_shared_heads, n_selected_heads,
                 dropout, seq_len, routing_mode="dynamic", kq_same=True):
        super().__init__()
        self.d_model = d_model
        
        # Transformer blocks with MoH attention
        self.blocks = get_clones(
            RouterTransformerLayer(
                d_model=d_model,
                d_feature=d_feature,
                d_ff=d_ff,
                dropout=dropout,
                n_heads=n_heads,
                n_shared_heads=n_shared_heads,
                n_selected_heads=n_selected_heads,
                seq_len=seq_len,
                routing_mode=routing_mode,
                kq_same=kq_same
            ),
            n_blocks
        )
        
        # Position embedding
        self.position_emb = CosinePositionalEmbedding(d_model=d_model, max_len=seq_len)
        
    def forward(self, q_embed_data, qa_embed_data, diff=None, response=None):
        # Add positional embeddings
        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb
        
        x = q_embed_data
        y = qa_embed_data
        
        # Track balance loss across all blocks
        total_balance_loss = 0
        
        # Pass through transformer blocks with proper masking
        for block in self.blocks:
            x, balance_loss = block(mask=0, query=x, key=x, values=y, diff=diff, response=response)
            total_balance_loss += balance_loss
            
        return x, total_balance_loss

class RouterTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, dropout, n_heads, 
                 n_shared_heads, n_selected_heads,
                 seq_len, routing_mode="dynamic", kq_same=True):
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
        
    def forward(self, mask, query, key, values, diff=None, response=None):
        # Create causal attention mask to prevent looking at future positions
        seqlen = query.size(1)
        nopeek_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool()
        src_mask = (~nopeek_mask).to(query.device)
        
        # Apply MoH attention with dynamic routing
        attn_output = self.attn(query, key, values, src_mask)
        x = self.layer_norm1(query + self.dropout1(attn_output))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_output))
        
        return x, self.attn.get_balance_loss() 