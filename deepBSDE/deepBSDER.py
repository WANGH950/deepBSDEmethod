import time
import torch
import torch.nn as nn
from equations import Equation

class deepBSDE(nn.Module):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, disc:nn.ModuleList, model_params:dict) -> None:
        super(deepBSDE,self).__init__()
        self.equation = equation
        self.result = result
        self.grad = grad
        self.disc = disc
        self.n = model_params['n']
        self.x = model_params['x']
        self.t = model_params['t']
        self.T = model_params['T']
        self.N = model_params['N']
    
    def forward(self, batch_size):
        return 0, 0

class DeepBSDER(deepBSDE):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, disc:nn.ModuleList, model_params:dict) -> None:
        super(DeepBSDER,self).__init__(equation,result,grad,disc,model_params)

    def forward(self, batch_size):
        delta_t = (self.T - self.t) / self.N
        process_N, process_X, discrete_t, delta_B = self.equation.get_positions(self.n, self.x, self.t, self.T, self.N, batch_size)
        
        alpha = self.equation.alpha(process_N)
        beta = self.equation.beta(process_N)
        u = self.result(process_N[0], process_X[0])
        for i in range(self.N):
            grad_u = self.grad[i](process_N[i], process_X[i])
            grad_bmm = torch.bmm(grad_u.unsqueeze(1), delta_B[i].unsqueeze(-1)).squeeze(-1)
            disc_u = self.disc[i](process_N[i], process_X[i])
            delta_N = (process_N[i+1] - process_N[i]).int()
            f = self.equation.f(discrete_t[i], process_N[i], process_X[i], u, grad_u)
            u = u - f * delta_t + grad_bmm - (alpha[i]*disc_u[:,0:1] + beta[i]*disc_u[:,1:2])*delta_t
            u[delta_N>0] = u[delta_N>0] + disc_u[:,0:1][delta_N>0]
            u[delta_N<0] = u[delta_N<0] + disc_u[:,1:2][delta_N<0]
        g = self.equation.g(process_N[self.N],process_X[self.N])
        return u, g
    
class IDeepBSDER(deepBSDE):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, disc:nn.ModuleList, model_params:dict) -> None:
        super(IDeepBSDER,self).__init__(equation,result,grad,disc,model_params)

    def forward(self, batch_size):
        delta_t = (self.T - self.t) / self.N
        process_N, process_X, discrete_t, delta_B = self.equation.get_positions(self.n, self.x, self.t, self.T, self.N, batch_size)
        
        g = self.equation.g(process_N[self.N],process_X[self.N])
        for i in range(self.N):
            j = self.N-i-1
            grad_u = self.grad[j](process_N[j], process_X[j])
            grad_bmm = torch.bmm(grad_u.unsqueeze(1), delta_B[j].unsqueeze(-1)).squeeze(-1)

            f = self.equation.f(discrete_t[j+1], process_N[j+1], None, g, None)
            g = g + f * delta_t - grad_bmm

        u = self.result(process_N[0], process_X[0])
        return u, g


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super(MSELoss,self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, input, target):
        return self.loss(input,target)
    

def train(model:deepBSDE, train_params:dict):
    epoch = train_params['epoch']
    batch_size = train_params['batch_size']
    lr = train_params['learning_rate']

    change_lr = train_params['change_lr']
    lr_change = train_params['lr_change']

    n = torch.tensor(train_params['n']).float()
    x = torch.tensor(train_params['x']).float()

    # init criterion
    criterion = MSELoss()

    # init Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    loss_values = torch.ones(epoch)
    result_values = torch.ones(epoch)
    # start training
    start = time.time()
    for i in range(epoch):
        # change learning rate or not?
        if change_lr and i == int(epoch/2):
            for param_grop in optimizer.param_groups:
                param_grop['lr'] = lr_change
        model.train()
        optimizer.zero_grad()
        u,g = model(batch_size)
        loss = criterion(u,g)
        loss.backward()
        optimizer.step()

        model.eval()
        loss_values[i] = loss.item()
        result_values[i] = model.result(n,x).detach()

        print('\r%5d/{}|{}{}|{:.2f}s  [Loss: %e, Result: %7.5f, disc1: %7.5f, disc2: %7.5f]'.format(
            epoch,
            "#"*int((i+1)/epoch*50),
            " "*(50-int((i+1)/epoch*50)),
            time.time() - start) %
            (i+1,
            loss_values[i],
            result_values[i],
            model.disc[0](n,x)[0,0],
            model.disc[0](n,x)[0,1]), end = ' ', flush=True)
    print("\nTraining has been completed.")
    return loss_values, result_values