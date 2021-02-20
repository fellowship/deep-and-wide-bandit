import torch
import numpy as np

class ContextualBandit():
    
    def __init__(self, device, model, optimizer, loss_func, algorithm):
        self.device = device
        self.model = model.to(device)
        self.context_dim = model.context_dim
        self.z_dim = model.z_dim
        self.action_dim = model.action_dim
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.algorithm = algorithm
        self.reset()
        
    def reset(self):
        self.dataset = []
    
    def observe_data(self, context_source):
        x_t = np.array(context_source)
        assert x_t.shape == (self.action_dim, self.context_dim)
        x_t_tensor = torch.tensor(x_t).float().to(self.device)
        return x_t_tensor 
    
    def get_reward(self, reward_source, a_t):
        r_t = float(reward_source[a_t])
        return r_t

    def run(self, context_source, reward_source):
        x_t = self.observe_data(context_source) # x_t torch.tensor size=(action_dim, context_dim)
        with torch.no_grad():
            z_t = self.model.get_z(x_t) # z_t torch.tensor size=(action_dim, z_dim)
        a_t = self.algorithm.select_action(z_t) # a_t int range (0, action_dim)
        r_t = self.get_reward(reward_source, a_t) # r_t float either from an online simulation or from a reward vertor(size=action_dim)  
        data = (x_t[a_t], z_t[a_t], torch.tensor(a_t).to(self.device), torch.tensor(r_t).to(self.device))
        if len(self.dataset) == 0:
            self.dataset = torch.hstack(data).unsqueeze(dim=0)
        else:
            self.dataset = torch.vstack((self.dataset, torch.hstack(data)))
        return a_t
    
    def train(self, start_index=0, batch_size=16, num_epoch=100):
        # prepare dataset and dataloader for training
        train_dataset = BanditDataset(self.dataset[start_index:], self.context_dim, self.z_dim)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # train num_epoch epoches
        for i in range(num_epoch):
            for data_batch in train_dataloader:
                contexts, _, actions, rewards = data_batch
                contexts = contexts.float().to(self.device)
                actions = actions.long().to(self.device)
                rewards = rewards.float().to(self.device)
                outputs = self.model(contexts)
                pred_rewards = outputs[range(outputs.shape[0]),actions]
                loss = self.loss_func(pred_rewards, rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()                
        # update algorithm's parameters after training
        with torch.no_grad():            
            upd_z = self.model.get_z(self.dataset[:,:self.context_dim])
        self.dataset[:,self.context_dim:(self.context_dim+self.z_dim)] = upd_z
    
class BanditDataset(torch.utils.data.Dataset):
    
    def __init__(self, raw_dataset, context_dim, z_dim):
        self.dataset = raw_dataset
        self.context_dim = context_dim
        self.z_dim = z_dim
    def __getitem__(self, index):
        data = self.dataset[index]
        c = data[:self.context_dim]
        z = data[self.context_dim:self.context_dim+self.z_dim]
        a = data[-2]
        r = data[-1]
        return c, z, a, r
    
    def __len__(self):
        return len(self.dataset)
    