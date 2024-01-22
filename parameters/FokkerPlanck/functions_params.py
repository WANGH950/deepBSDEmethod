import torch
import numpy as np

def lamb_bkwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(40-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(30-n) + torch.sign(torch.relu(n-14)*torch.relu(100-n))*torch.abs(100-n)/10

def alpha_bkwd(n):
    return lamb_bkwd(n)

def mu_bkwd(n):
    return torch.sign(torch.relu(10-n))*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(n) + torch.sign(torch.relu(n-29))*5

def beta_bkwd(n):
    return mu_bkwd(n)


def g_bkwd(n, x):
    dim = x.shape[1]
    u0 = torch.exp(-10*torch.norm(x,dim=1,keepdim=True)**2/(n+1))*torch.sqrt((10/(n+1)/torch.tensor(np.pi))**dim)
    return u0.float()*(n%3 == 1)


def g_bkwdc(n, x):
    dim = x.shape[1]
    u0 = (torch.exp(-torch.norm(x-1,dim=1,keepdim=True)**2*(n+1)) + torch.exp(-torch.norm(x+1,dim=1,keepdim=True)**2*(n+1)))*torch.sqrt(((n+1)/torch.tensor(np.pi))**dim)/2
    return u0.float()*(n%2)


def f_bkwd(t,n,x,u,grad):
    return 0


def lamb_fwd(n):
    return torch.sign(torch.relu(5-n))*torch.abs(20-n)**(1/3) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*torch.sqrt(torch.abs(30-n)) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*torch.abs(20-n) + torch.sign(torch.relu(n-14)*torch.relu(50-n))*torch.abs(50-n)/10

def mu_fwd(n):
    return torch.sign(torch.relu(10-n))*1.5*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*torch.abs(n-5) + torch.sign(torch.relu(n-19)*torch.relu(30-n))*torch.sqrt(torch.abs(n-10)) + torch.sign(torch.relu(n-29))*5

def alpha_fwd(n):
    return mu_fwd(n+1)

def beta_fwd(n):
    res = lamb_fwd(n-1)
    res[n==0] = 0
    return res

def g_fwd(n, x):
    dim = x.shape[1]
    g = torch.exp(-torch.norm(x,dim=1,keepdim=True)**2 / (n+1))
    return g.float()*(n%2==0)

def f_fwd(t,n,x,u,grad):
    param = alpha_fwd(n) + beta_fwd(n) - alpha_fwd(n-1) - beta_fwd(n+1)
    return param*u

def lamb_bkwd_high_dim(n):
    res = torch.sign(torch.relu(5-n))*(20-n) + torch.sign(torch.relu(n-4)*torch.relu(10-n))*(30-n) + torch.sign(torch.relu(n-9)*torch.relu(15-n))*(40-n) + torch.sign(torch.relu(n-14)*torch.relu(50-n))*(50-n)
    return res

def alpha_bkwd_high_dim(n):
    return lamb_bkwd_high_dim(n)

def mu_bkwd_high_dim(n):
    res = torch.sign(torch.relu(10-n))*2*n + torch.sign(torch.relu(n-9)*torch.relu(20-n))*n + torch.sign(torch.relu(n-19)*torch.relu(30-n))*n/2 + torch.sign(torch.relu(n-29))*7.5
    return res

def beta_bkwd_high_dim(n):
    return mu_bkwd_high_dim(n)

def g_bkwd_high_dim(n, x):
    g = 1 - (0.5 + 0.49*(torch.sin(torch.sum(x,dim=1,keepdim=True)*n)*(n%2) + torch.cos(torch.sum(x,dim=1,keepdim=True)*n)*(1-(n%2))))
    return g.float()

def f_bkwd_high_dim(t,n,x,u,grad):
    return u - u**2