import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        self.end_conv1 = nn.Conv1d(in_channels=d_model, out_channels=int(d_model / 2), kernel_size=1, bias=True)
        self.end_conv2 = nn.Conv1d(in_channels=int(d_model / 2), out_channels=int(d_model / 4), kernel_size=1,
                                   bias=True)
        self.projection1 = nn.Linear(int(d_model / 2), int(d_model / 4), bias=True)
        self.projection2 = nn.Linear(int(d_model / 8), c_out, bias=True)
        self.filter = nn.Linear(self.pred_len, c_out, bias=True)
        self.lstm_layer = nn.LSTM(input_size=int(d_model / 4), hidden_size=d_model, num_layers=2,
                                  bias=True, batch_first=True, dropout=0.5, bidirectional=False)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None): #   x_enc 和 x_dec去掉了SOC维度
        enc_out = self.enc_embedding(x_enc, x_mark_enc) #x_enc:[32,96,3] enc_out:[32,96,512]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = dec_out.permute(0, 2, 1)  #张量维度更换，第0维不变，1维和2维互换
        dec_out = self.end_conv1(dec_out)   #卷积降维512-<256
        dec_out = self.end_conv2(dec_out)   #卷积将为256->128
        dec_out = dec_out.permute(0, 2, 1)
        dec_out, (h_n, c_n) = self.lstm_layer(dec_out)  #维度恢复512  lstm out: [batch, label+pred, hidden_size]
        dec_out = self.projection(dec_out)  #取均值降维

        '''dec_out = self.projection1(dec_out.permute(0, 2, 1))
        
        dec_out = self.projection2(dec_out.permute(0, 2, 1))'''
        # [batch, label_len+pred_len, 1]
        dec_out = dec_out[:, -self.pred_len:, :]
        # dec_out = self.filter(dec_out.permute(0, 2, 1))

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # Embedding编码, [batch, seq_len, d_model=512]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # enc_out: [batch, seq_len/2, d_model]

        # decoder输入Embedding操作，和encoder一样
        # x_dec: [batch, label_len+pred_len, feature_size]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # [batch, label_len+pred_len, d_model]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # [batch, label_len+pred_len, d_model]
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        # 取后面pred_len个数据
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
