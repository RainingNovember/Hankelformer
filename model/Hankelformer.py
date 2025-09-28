import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

def create_hankel_matrix(x, window_size):
    batch_size, seq_len, num_features = x.shape
    hankel_matrices = []
    
    for i in range(seq_len - window_size + 1):
        hankel_slice = x[:, i:i+window_size, :]
        hankel_matrices.append(hankel_slice)
    
    return torch.stack(hankel_matrices, dim=1)


class HankelMLP(nn.Module):
    def __init__(self, input_size, hidden_size, window_size):
        super(HankelMLP, self).__init__()
        self.window_size = window_size
        self.input_size = input_size
        
        self.mlp = nn.Sequential(
            nn.Linear(window_size * input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape
        
        x_hankel = create_hankel_matrix(x, self.window_size)
        # x_hankel shape: [batch_size, new_seq_len, window_size, input_size]
        
        new_seq_len = seq_len - self.window_size + 1
        
        x_reshaped = x_hankel.reshape(batch_size, new_seq_len, -1)
        # x_reshaped shape: [batch_size, new_seq_len, window_size * input_size]
        
        x_processed = self.mlp(x_reshaped)
        # x_processed shape: [batch_size, new_seq_len, input_size]
        
        padding = torch.zeros(batch_size, self.window_size - 1, self.input_size, device=x.device)
        x_padded = torch.cat([x_processed, padding], dim=1)
        # x_padded shape: [batch_size, seq_len, input_size]
        
        return x_padded



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.hankel_mlp = HankelMLP(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            # window_size=1
            # window_size=1
            window_size=configs.window_size
        )

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.fusion_layer = nn.Linear(configs.d_model * 2, configs.d_model)

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def encode(self, x, x_mark):
        enc_out = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        global_repr = enc_out.mean(dim=1)  # [batch_size, d_model]
        
        # contrastive_repr = self.contrastive_head(global_repr)  # [batch_size, contrastive_dim]
        
        return enc_out, global_repr

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc_processed = self.hankel_mlp(x_enc)

        enc_out, contrastive_repr = self.encode(x_enc, x_mark_enc)
        enc_out_processed, contrastive_repr_processed = self.encode(x_enc_processed, x_mark_enc)

        # print("enc_out shape",enc_out.shape) # torch.Size([16, 325, 512])
        # print("enc_out_processed shape",enc_out_processed.shape) # torch.Size([16,
        # print("contrastive_repr shape",contrastive_repr.shape) # torch.Size([16, 325, 512])
        # print("contrastive_repr_processed shape",contrastive_repr_processed.shape) # torch.Size([16, 325, 512])

        fused_enc_out = self.fusion_layer(torch.cat([enc_out, enc_out_processed], dim=-1))

        # print("fused_enc_out shape",fused_enc_out.shape) # torch.Size([16, 325, 512])


        dec_out = self.projector(fused_enc_out).permute(0, 2, 1)[:, :, :x_enc.shape[2]]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, contrastive_repr, contrastive_repr_processed

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, contrastive_repr, contrastive_repr_processed = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return dec_out[:, -self.pred_len:, :], contrastive_repr, contrastive_repr_processed
