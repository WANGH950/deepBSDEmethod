'''
An example of code that uses the model. 
You can define any of these components yourself.
'''
from util import ParameterResult, DMLP
import parameters.FokkerPlanck.functions_params as funp
import deepBSDE.deepBSDER as dbr
import equations as eq
import torch.nn as nn
import torch
import json

# load parameters
path = './parameters/FokkerPlanck/Backward.json'
f = open(path,'r')
params = json.load(f)

dis_params = params['High_dimensional']
equation_params = dis_params['equation_params']
model_params = dis_params['model_params']
train_params = dis_params['train_params']

# Instantiate the PDE
equation = eq.FokkerPlanck(
    parameters=equation_params,
    alpha=funp.alpha_bkwd_high_dim,
    beta=funp.beta_bkwd_high_dim,
    g=funp.g_bkwd_high_dim,
    f=funp.f_bkwd_high_dim
)

# Instantiate the basis function
dim = equation_params['dim']
N = model_params['N']
model_params['x'] = [[0 for _ in range(dim)],[0 for _ in range(dim)]]
train_params['x'] = [[0 for _ in range(dim)]]
result = ParameterResult(
    dim=1,
    min=-0.1,
    max=0.1
)
grad = nn.ModuleList([
    ParameterResult(
        dim=dim,
        min=-1/torch.sqrt(torch.tensor(dim)),
        max=1/torch.sqrt(torch.tensor(dim))
    )
]+[
    DMLP(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=128,
        layer_num=3,
        batch_norm=False
    ) for _ in range(N-1)
])
disc = nn.ModuleList([
    ParameterResult(
        dim=2,
        min=-1/torch.sqrt(torch.tensor(2)),
        max=1/torch.sqrt(torch.tensor(2))
    )
]+[
    DMLP(
        input_dim=dim,
        output_dim=2,
        hidden_dim=128,
        layer_num=3
    ) for _ in range(N-1)
])

# Assemble the model
model = dbr.DeepBSDER(
    equation=equation,
    result=result,
    grad=grad,
    disc=disc,
    model_params=model_params
)

# Train the model
loss_values, result_values = dbr.train(
    model=model,
    train_params=train_params
)