from turtle import forward
import torch
from torch import nn
from src.models.const import ModelKey

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.modelkey = ModelKey()
        
    def forward(self):
        raise NotImplementedError
        
    @torch.no_grad()
    def inference(self):
        raise NotImplementedError