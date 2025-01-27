import os
import math
import time 
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttension(nn.Module): 
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # to check emdb dims can be divide in different heads equally for concat.
    # key, query, value projections for all heads, but in a batch
    self.c_atten=nn.linear(config.n_embd , 3* config.n_embd) 
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
  
  def forward(self,x):
    B,T,C= x.size() ## batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    qkv=self.c_atten(x)
    #as a eg x= 16,1024,768(B,T,C) qkv=16,1024,2304
    q,k,v=qkv.split(self.n_embd,dim=2) 
    # split back into 16,1024,768 dims.
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs), After transposing, the shape changes from (B, T, n_head, C // n_head) to (B, n_head, T, C // n_head)
    y=F.scaled_dot_product_attention(q,k,v, is_casual=True) #Flash attension
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    # output projection, Feed forward
    y = self.c_proj(y)
    return y

class MLP(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.c_fc= nn.Linear(config.n_embd,4*config.n_embd)
    self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
    self.nonlin=nn.GELU()
    self.c_proj.NANOGPT_SCALE_INIT = 1
  # simply making MLP , increasing dimensions for NN to think about info received in attension ,back to embd for logits 768-->3072-->3072

  def Forward(self,x):
    x=self.c_fc(x)
    x=self.gelu(x)
    x=self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.ln_1=nn.LayerNorm(config.n_embd) # normalizing the weights  
    self.attn=CasualSelfAttension(config) # self-attension block
    self.ln_2=nn.LayerNorm(config.n_embd) # normalizing 
    self.mlp= MLP(config)
  
  def forward(self,x):
    x= x+self.attn(self.ln_1(x)) # residual connection
    x= x+self.mlp(self.ln_2(x)) # residual connection , gradients can flow directly to x


@dataclass

class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    # head size= 64

class GPT(nn.Module):
  def __init__(self,config):
    super().__init__()

    self.transformer

