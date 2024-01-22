import torch

class Equation():
    def __init__(self, g, f) -> None:
        self.g = g
        self.f = f
    
    def get_positions():
        return None

class FokkerPlanck(Equation):
    def __init__(self, parameters, alpha, beta, g, f):
        super(FokkerPlanck,self).__init__(g,f)
        self.D0 = parameters['D0']
        self.dim = parameters['dim']
        self.nmin = parameters['nmin']
        self.a = parameters['alpha']
        self.alpha = alpha
        self.beta = beta
    
    def D(self, n):
        return self.D0 / (n+self.nmin)**self.a

    def next_position(self, pre_n, pre_x, delta_B, delta_t):
        alpha = self.alpha(pre_n)*delta_t
        beta = self.beta(pre_n)*delta_t
        rand = torch.rand_like(pre_n)
        delta_n = (rand<alpha).float() - (rand>1-beta).float()
        next_N = pre_n + delta_n
        next_X = pre_x + torch.sqrt(2*self.D(pre_n))*delta_B
        return next_N, next_X
    
    def get_positions(self, n, x, t, T, N, size):
        delta_t = torch.tensor((T-t)/N)
        n = torch.tensor(n)
        x = torch.tensor(x)
        discrete_t = torch.ones([N+1,size,1])*torch.linspace(t,T,N+1).reshape([N+1,1,1])
        process_N = torch.ones([N+1,size,1])*torch.randint(n[0],n[1]+1,[size,1])
        process_X = torch.ones([N+1,size,self.dim])*torch.rand([size,self.dim])*(x[1]-x[0])+x[0]
        delta_B = torch.randn([N,size,self.dim])*torch.sqrt(delta_t)
        for i in range(N):
            process_N[i+1], process_X[i+1] = self.next_position(process_N[i],process_X[i],delta_B[i],delta_t)
        return process_N, process_X, discrete_t, delta_B

class FeynmanKac(FokkerPlanck):
    def __init__(self, parameters, alpha, beta, g, f, U):
        super(FeynmanKac,self).__init__(parameters,alpha,beta,g,f)
        self.U = U
    
    def get_positions(self, n, x, p, t, T, N, size):
        p = torch.tensor(p)
        discrete_p = torch.ones([N+1,size,1])*torch.rand([size,1])*(p[1]-p[0])+p[0]
        process_N, process_X, discrete_t, delta_B = super().get_positions(n, x, t, T, N, size)
        return process_N, process_X, discrete_p, discrete_t, delta_B