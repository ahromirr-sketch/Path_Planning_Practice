import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        hidden_size = 256
        
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh() 
        )
        
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def act(self, state):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), value.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(-1), dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.action_log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MSELoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            # 💡 수정: 병렬 환경이므로 이미 (8, 11) 형태입니다. unsqueeze 제거!
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_value = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(state_value.squeeze(-1))
        
        return action.cpu().numpy() # (8, 2) 형태로 반환되어 환경 8개로 흩어짐!

    def update(self):
        rewards = []
        # 💡 수정: 드론이 8대니까 미래 점수 계산도 8대분을 동시에 계산합니다!
        discounted_reward = torch.zeros(len(self.buffer.is_terminals[0])).to(device)
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            reward = reward.to(device)
            is_terminal = is_terminal.to(device)
            # 죽은 드론(is_terminal=1)은 (1 - 1) = 0이 곱해져서 과거 점수 연동이 끊어짐
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - is_terminal))
            rewards.insert(0, discounted_reward)
            
        # (N스텝, 8대) 형태의 2D 행렬을 GPU가 씹어먹기 좋게 (N*8) 1D 행렬로 쭉 폅니다.
        rewards = torch.stack(rewards).flatten().detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 상태와 행동들도 전부 (N*8) 크기의 행렬로 쭉쭉 펴줍니다. (이게 병렬 텐서의 핵심!)
        old_states = torch.stack(self.buffer.states).view(-1, 11).detach()
        old_actions = torch.stack(self.buffer.actions).view(-1, 2).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).flatten().detach()
        old_values = torch.stack(self.buffer.values).flatten().detach()

        advantages = rewards - old_values

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MSELoss(state_values, rewards)
            entropy_bonus = -0.01 * dist_entropy.mean()
            
            loss = actor_loss + critic_loss + entropy_bonus
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()