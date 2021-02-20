import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class LinUCB():
    
    def __init__(self, device, z_dim, action_dim, delta=0.05):
        self.device = device
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.alpha = 1.0 + np.sqrt(np.log(2 / delta) / 2)
        self.reset()
    
    def reset(self):
        self.A = [torch.eye(self.z_dim).to(self.device) for i in range(self.action_dim)]
        self.b = [torch.zeros(self.z_dim).to(self.device) for i in range(self.action_dim)]
    
    def select_action(self, z_t):
        theta = torch.zeros((self.action_dim, self.z_dim)).to(self.device)
        ucb = torch.zeros(self.action_dim).to(self.device)
        for a in range(self.action_dim):
            A_inv = torch.inverse(self.A[a])
            theta[a] = torch.matmul(A_inv, self.b[a])
            ucb[a] = torch.dot(z_t[a], theta[a]) + self.alpha * torch.sqrt(torch.dot(torch.matmul(z_t[a], A_inv), z_t[a]))
        
        return torch.argmax(ucb).item()
    
    def update_one(self, data):
        assert len(data.shape) == 1
        z = data[(-self.z_dim-2):-2]
        a = data[-2]
        r = data[-1]
        self.A[int(a)] += torch.matmul(z.reshape((self.z_dim, 1)), z.reshape((1, self.z_dim)))
        self.b[int(a)] += r * z

    def update_all(self, data):
        for a in data[:,-2].unique():
            data_a = data[data[:,-2]==a,:]
            z = data_a[:,(-self.z_dim-2):-2]
            r = data_a[:,-1]
            self.A[int(a)] = torch.matmul(z.T, z) + torch.eye(self.z_dim).to(data.device)
            self.b[int(a)] = torch.matmul(z.T, r)


from scipy.stats import invgamma

class TS():
    
    def __init__(self, z_dim, action_dim, lambda_=0.25):
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        self.cov = [1/self.lambda_*torch.eye(self.z_dim) for i in range(self.action_dim)]
        self.mu = [torch.zeros(self.z_dim) for i in range(self.action_dim)]
        self.a0 = 6.0
        self.b0 = 6.0
        self.a = [self.a0 for i in range(self.action_dim)]
        self.b = [self.b0 for i in range(self.action_dim)]
    
    def select_action(self, z_t):
        z_t_np = z_t.cpu().numpy()
        probs = []
        for i in range(self.action_dim):
            b = self.b[i]
            a = self.a[i]
            cov = self.cov[i].cpu().numpy()
            cov = b * invgamma.rvs(a) * cov
            mu = self.mu[i].cpu().numpy()
            beta = np.random.multivariate_normal(mu, cov)
            probs.append(np.dot(beta, z_t_np[i]))
        return np.argmax(probs)

    def update_all(self, data):
        # data torch tensor shape=(len, (context_dim + z_dim + 1(for action) + 1(for reward)))
        for a in data[:,-2].unique():
            data_a = data[data[:,-2]==a,:]
            zs = data_a[:,-self.z_dim-2:-2]
            rs = data_a[:,-1]
            s = torch.matmul(zs.T,zs)
            precision = s + self.lambda_*torch.eye(self.z_dim).to(data.device)
            self.cov[int(a)] = torch.inverse(precision)
            self.mu[int(a)] = torch.matmul(self.cov[int(a)], torch.matmul(zs.T, rs))
            self.a[int(a)] = self.a0 + len(data_a)/2.0
            self.b[int(a)] = self.b0 + 0.5 * torch.dot(rs, rs).item() - 0.5 * torch.dot(self.mu[int(a)], torch.matmul(precision, self.mu[int(a)])).item()