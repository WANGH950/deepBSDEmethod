import torch
import numpy as np

def lamb_bkwd(n):
    return torch.sign(torch.relu(5-n))*(20-n) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(40-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*(30-n) + torch.sign(torch.relu(n-14)*torch.relu(100-n))*(100-n)/10

def alpha_bkwd(n):
    return lamb_bkwd(n)

def mu_bkwd(n):
    return torch.sign(torch.relu(10-n))*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(n) + torch.sign(torch.relu(n-29))*5

def beta_bkwd(n):
    return mu_bkwd(n)

def lamb_fwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n)**(1/3) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(30-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(20-n) + torch.sign(torch.relu(n-14)*torch.relu(50-n))*torch.abs(50-n)/10

def mu_fwd(n):
    return torch.sign(torch.relu(10-n))*1.5*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(torch.abs(n-10)) + torch.sign(torch.relu(n-29))*5

def alpha_fwd(n):
    return mu_fwd(n+1)

def beta_fwd(n):
    res = lamb_fwd(n-1)
    res[n==0] = 0
    return res

# functional
def U_positional_func(x):
    return torch.norm(x,dim=2,keepdim=True)**2

def U_occupation_time(x):
    return torch.relu(torch.sign(x[:,:,:1]))

def g_fwd(n, x):
    u_rel = torch.exp(-torch.norm(x,dim=1,keepdim=True)**2 / (n+1))
    return u_rel*(n%2) + 1j*0

def f_fwd(t,n,x,p,u,grad):
    param = alpha_fwd(n) + beta_fwd(n) - alpha_fwd(n-1) - beta_fwd(n+1)
    return param*u

def g_bkwd(n, x):
    return torch.ones_like(n) + 1j*0

def f_bkwd(t,n,x,p,u,grad):
    return 0