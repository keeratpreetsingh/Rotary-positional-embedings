import torch
import torch.nn as nn 
class Rope(nn.Module):
    def __init__(self,context_lenght,dim):
        super().__init__()
        self.no_of_pairs=dim//2
        self.freq=1/10000**(torch.arange(0,self.no_of_pairs)/self.no_of_pairs)
        self.pos=torch.arange(context_lenght)
        self.angles=torch.einsum('i,j->ij',self.pos,self.freq)
        self.cos=torch.cos(self.angles)
        self.sin=torch.sin(self.angles)
    def forward(self,x):
        b,s,d=x.shape
        x_=x.view(b,s,d//2,2)
        sin=self.sin.unsqueeze(0).unsqueeze(-1)
        cos=self.cos.unsqueeze(0).unsqueeze(-1)
        x_rotate=torch.zeros_like(x_)
        x_rotate[...,0]=x_[...,0]*cos - x_[...,1]*sin
        x_rotate[...,1]=x_[...,0]*sin + x_[...,1]*cos
        x_rotate=x.view(b,s,d)
        return x_rotate
