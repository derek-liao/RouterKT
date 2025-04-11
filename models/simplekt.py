import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss
# from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, CosinePositionalEmbedding, MultiHeadAttentionWithContextDistance, MultiHeadAttention
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention, MultiHeadAttentionWithContextDistance, CosinePositionalEmbedding
from .rpe import SinusoidalPositionalEmbeddings
import numpy as np
# copy
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class SimpleKT(Module):
    def __init__(self, device, num_skills, num_questions, seq_len, bincounts,
                 embedding_size, num_blocks, dropout, d_ff=256, 
                 num_attn_heads=8, separate_qa=False, l2=1e-5,
                 final_fc_dim=512, final_fc_dim2=256, de_type="none_0", model_type="simplekt",
                 choose_enc="g", emb_path="", pretrain_dim=768, kq_same=True):
        super().__init__()
        self.device = device
        self.model_name = "simplekt"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.dropout = dropout
        self.l2 = l2
        self.separate_qa = separate_qa
        self.de = de_type.split('_')[0]
        self.token_num = int(de_type.split('_')[1])
        self.choose_enc = choose_enc
        self.seq_len = seq_len
        
        # Position embedding - use different types based on de_type
        if self.de.startswith("basic"):
            self.position_emb = Embedding(seq_len + 1, embedding_size, padding_idx=0)
        else:
            self.position_emb = CosinePositionalEmbedding(d_model=embedding_size, max_len=seq_len)
        
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
            
        # Position and difficulty embeddings
        if self.de.startswith(("sde", "alibi-sde", "rotary-sde")):
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(self.token_num+1, embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
        elif self.de.startswith("random"):
            self.diff_emb = Embedding(self.token_num+1, embedding_size)
            
        # Main architecture
        self.model = Architecture(
            device=device,
            n_question=num_skills,
            n_blocks=num_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=embedding_size,
            d_feature=embedding_size // num_attn_heads,
            d_ff=d_ff,
            seq_len=seq_len,
            de_type=de_type,
            bincounts=bincounts,
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
        pos = feed_dict["position"][:, :-1] if "position" in feed_dict else None
        diff = feed_dict["sdiff"] if "sdiff" in feed_dict else None
        if diff is not None:
            diff = (diff*(feed_dict["responses"]>-1).int()).long()
        
        # Get base embeddings for previous timesteps
        q_embed_data = self.q_embed(q)  # c_c
        if self.separate_qa:
            qa_data = q + self.num_skills * masked_r
            qa_embed_data = self.qa_embed(qa_data)
        else:
            masked_r = r * (r > -1).long()
            qa_embed_data = q_embed_data + self.qa_embed(masked_r)  # c_c + g_r
            
        # Add difficulty embeddings if specified
        if self.de.startswith(("sde", "random", "alibi-sde", "rotary-sde")) and diff is not None:
            q_embed_data = q_embed_data + self.diff_emb(diff[:, :-1]).float()
            
        # Add position embeddings based on encoding type
        if self.de.startswith(("basic", "sde", "random")) and pos is not None:
            if self.de.startswith("basic"):
                posemb = self.position_emb(pos)  # For basic, use learned embeddings
            else:
                posemb = self.position_emb(q_embed_data)  # For other types, use cosine
            q_embed_data = q_embed_data + posemb
            qa_embed_data = qa_embed_data + posemb
            
        # Add question difficulty if using question IDs
        if self.num_questions > 0 and "questions" in feed_dict:
            pid_data = feed_dict["questions"][:, :-1]
            q_embed_diff_data = self.q_embed_diff(q)  # d_c
            pid_embed_data = self.difficult_param(pid_data)  # μ_q
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # c_c + μ_q * d_c
            
            qa_embed_diff_data = self.qa_embed_diff(q + self.num_skills * masked_r)  # f_(c,r)
            qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            
        # Get embeddings for current skills to predict
        qnext_embed_data = self.q_embed(qnext)
        if self.de.startswith(("sde", "random", "alibi-sde", "rotary-sde")) and diff is not None:
            qnext_embed_data = qnext_embed_data + self.diff_emb(diff[:, 1:]).float()
            
        # Determine encoding type for transformer
        enc = None
        if self.de.startswith(("alibi", "rotary", "basic", "relative")):
            enc = diff
            
        # Pass through transformer
        d_output = self.model(q_embed_data, qa_embed_data, enc, r)
        
        # Final prediction
        concat_q = torch.cat([d_output, qnext_embed_data], dim=-1)
        output = torch.sigmoid(self.out(concat_q)).squeeze(-1)
        
        out_dict = {
            "pred": output,
            "true": feed_dict["responses"][:, 1:].float()
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss + self.l2 * torch.norm(self.difficult_param.weight), len(pred[mask]), true[mask].sum().item()

class Architecture(Module):
    def __init__(self, device, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, seq_len, de_type, bincounts, kq_same):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.de_type = de_type
        
        # Transformer blocks
        self.blocks = get_clones(
            TransformerLayer(
                device=device,
                d_model=d_model,
                d_feature=d_feature,
                d_ff=d_ff,
                dropout=dropout,
                n_heads=n_heads,
                seq_len=seq_len,
                de_type=de_type,
                bincounts=bincounts,
                kq_same=kq_same
            ),
            n_blocks
        )
        
    def forward(self, q_embed_data, qa_embed_data, enc, r):
        x = q_embed_data
        y = qa_embed_data
        
        # Pass through transformer blocks with proper masking
        for block in self.blocks:
            x = block(query=x, key=x, values=y, diff=enc, response=r)
            
        return x

class TransformerLayer(Module):
    def __init__(self, device, d_model, d_feature, d_ff, dropout, n_heads, seq_len, de_type, bincounts, kq_same):
        super().__init__()
        self.device = device
        self.de_type = de_type
        
        # Multi-head attention
        if de_type.startswith("monotonic"):
            kq_same = False
            self.attn = MultiHeadAttentionWithContextDistance(
                d_model, d_feature, n_heads, dropout, kq_same=kq_same, 
                seq_len=seq_len, de_type=de_type, bincounts=bincounts
            )
        else:
            self.attn = MultiheadAttention(
                d_model, n_heads, dropout=dropout, 
                seq_len=seq_len, de_type=de_type, bincounts=bincounts
            )
        
        # Layer norm and dropout
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        
        # Feed forward network
        self.ffn = transformer_FFN(d_model, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, query, key, values, diff=None, response=None):
        # Create causal attention mask
        causal_mask = ~ut_mask(self.device, seq_len=key.shape[1])
        
        # Apply attention with appropriate encoding
        attn_emb, attn = self.attn(query, key, values, diff=diff, response=response, mask=causal_mask)
        attn_emb = self.dropout1(attn_emb)
        attn_emb = self.layer_norm1(query + attn_emb)
        
        # Feed forward
        ffn_output = self.ffn(attn_emb)
        ffn_output = self.dropout2(ffn_output)
        output = self.layer_norm2(attn_emb + ffn_output)
        
        return output
    
class RotaryEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        
        x1, x2 = x.chunk(2, dim=-1)
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        
        return torch.cat([rx1, rx2], dim=-1) 

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, score_mask=None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if score_mask is not None:
        scores += score_mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
