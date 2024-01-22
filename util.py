import torch
import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num = 2, batch_norm = False):
        super(MLP,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_norm = batch_norm
        if layer_num == 0:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            self.mlp = nn.Sequential(
                OrderedDict(
                [("hidden_layer_0",
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh()
                    )
                  )] + 
                [("hidden_layer_"+str(i+1),
                    nn.Sequential(
                        nn.Linear(hidden_dim,hidden_dim),
                        nn.Tanh()
                    )) for i in range(layer_num-1)] + 
                [("output_layer",nn.Linear(hidden_dim, output_dim))]
                )
            )
        if batch_norm:
            self.mlp = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                self.mlp
            )
    
    def forward(self, input):
        return self.mlp(input)

class MLPResult(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPResult,self).__init__(input_dim, output_dim, hidden_dim, layer_num, batch_norm)
    
    def forward(self, n, x):
        return super().forward(x)


class MLPGrad(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPGrad,self).__init__(input_dim+1, output_dim, hidden_dim, layer_num, batch_norm)
    
    def forward(self, n, x):
        return super().forward(torch.cat([n,x],dim=1))


class MLPComplex(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num=2, batch_norm=False):
        super(MLPComplex,self).__init__(input_dim, output_dim*2, hidden_dim, layer_num, batch_norm)
        self.output_dim = output_dim

    def forward(self, input):
        output = super().forward(input)
        return output[:,:self.output_dim] + 1j*output[:,self.output_dim:]


class MLPResultComplex(MLPComplex):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPResultComplex,self).__init__(input_dim, output_dim, hidden_dim, layer_num, batch_norm)
    
    def forward(self, n, x, p):
        return super().forward(p)


class MLPGradComplex(MLPComplex):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPGradComplex,self).__init__(input_dim+2, output_dim, hidden_dim, layer_num, batch_norm)

    def forward(self, n, x, p):
        return super().forward(torch.cat([n,x,p],dim=1))

class PositionalEncoding(nn.Module):
    def __init__(self, dim) -> None:
        super(PositionalEncoding,self).__init__()
        self.dim = dim
        self.wk = 1 / 10**(8*torch.arange(0,dim//2,1)/dim)
        self.scale_n = nn.Parameter(torch.randn(dim) / torch.sqrt(torch.tensor(dim)),requires_grad=True)

    def forward(self, n):
        res = torch.cat([torch.sin(n*self.wk),torch.cos(n*self.wk)],dim=1)
        return res*self.scale_n**2

class DMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num=3, batch_norm=False) -> None:
        super(DMLP,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_norm = batch_norm
        self.PE = PositionalEncoding(
            dim=hidden_dim
        )
        self.in_layer = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Tanh()
        )
        self.hidden_layer = nn.ModuleList(
            [(nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.Tanh()
            )) for _ in range(layer_num)]
        )
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,output_dim)
        )
        if batch_norm:
            self.in_layer = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                self.in_layer
            )

    def forward(self, n, x):
        n = self.PE(n)
        y = self.in_layer(x)
        for i in range(self.layer_num):
            y = self.hidden_layer[i](y+n)
        return self.out_layer(y)


class DMLPComplex(DMLP):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num=3, batch_norm=False):
        super(DMLPComplex,self).__init__(input_dim+1, output_dim*2, hidden_dim, layer_num, batch_norm)
        self.output_dim_rel = output_dim

    def forward(self, n, x, p):
        output = super().forward(n,torch.cat([x,p],dim=1))
        return output[:,:self.output_dim_rel] + 1j*output[:,self.output_dim_rel:]


class ParameterResult(nn.Module):
    def __init__(self, dim, min=-1, max=1) -> None:
        super(ParameterResult,self).__init__()
        self.dim = dim
        self.result = nn.Parameter(data=torch.rand(dim)*(max-min)+min,requires_grad=True)
    
    def forward(self, n, x):
        batch_size = n.shape[0]
        return torch.ones([batch_size,self.dim])*self.result