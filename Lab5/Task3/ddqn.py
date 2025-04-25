# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0) # /255 to let value strict in 0-1, avoid gradient explode


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            return obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked

class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        priority = (abs(error) + 1e-6) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)           
        else:
            self.buffer[self.pos] = transition       

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos] 

        probs = prios / prios.sum() 
        indices = np.random.choice(len(probs), batch_size, p=probs)

        # Importance‑sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()    

        transitions = [self.buffer[idx] for idx in indices]
        ########## END OF YOUR CODE (for Task 3) ########## 
        return transitions, indices, torch.tensor(weights, dtype=torch.float32)
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha            
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def __len__(self):       
        return len(self.buffer)    
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.preprocessor = AtariPreprocessor()
        self.state_dim = self.preprocessor.frame_stack
        self.num_actions = self.env.action_space.n
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.state_dim, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=args.lr, alpha=0.95, eps=1e-2)
        #self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay_steps = args.epsilon_decay_steps


        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        self.eval_final_reward = 0
        os.makedirs(self.save_dir, exist_ok=True)

        # for multi-step return
        self.n_steps = args.n_steps
        self.nstep_gamma = self.gamma ** self.n_steps 
        self.nstep_buffer = deque(maxlen=self.n_steps) 

    def select_action(self, state):
        if random.random() < self.epsilon: # explore actions
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device) # add a dim. -> (b, c, h, w)
        with torch.no_grad():
            q_values = self.q_net(state_tensor) # calculate q values
        return q_values.argmax().item() # use the action which have biggest value

    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            if step_count == 0:
                self.nstep_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action) # return observation, reward, whether it's done or not, debug info.
                done = terminated or truncated
                reward = np.clip(reward, -1, 1)

                next_state = self.preprocessor.step(next_obs)
                self.nstep_buffer.append( (state, action, reward, next_state, done) )

                # Multi-step return
                if len(self.nstep_buffer) == self.n_steps:
                    R, s_n, done_n = 0.0, next_state, done
                    for i, (_,_,r, s_i, d_i) in enumerate(reversed(self.nstep_buffer)):
                        R = r + (self.gamma * R * (1 - d_i))
                        if d_i: 
                            s_n, done_n = s_i, True
                            break
                    s_0, a_0, *_ = self.nstep_buffer[0]
                    max_p = self.memory.priorities[: len(self.memory)].max() if len(self.memory) else 1.0
                    self.memory.add((s_0, a_0, R, s_n, done_n), error=max_p)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                
                if self.env_count == 200000 or self.env_count == 400000 or self.env_count == 600000 or self.env_count == 800000 or self.env_count == 1000000 or self.env_count == 1200000:
                    model_path = os.path.join(self.save_dir, f"env_count_model{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved env_count {self.env_count} model to {model_path}")

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                self.eval_final_reward += eval_reward
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
            

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 

        decay_progress = min(1.0, self.env_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start - decay_progress * (self.epsilon_start - self.epsilon_min)
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch, indices, is_w = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
      
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Selects the predicted Q-values for the actions actually taken from the network's output.
        is_w = is_w.to(self.device)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True) 
            q_nexts      = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_targets    = rewards + (self.nstep_gamma * q_nexts * (1 - dones))
        td_errors = q_targets - q_values                 
        loss = (is_w * td_errors.pow(2)).mean() # MSE

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        beta_end, beta_frames = 1.0, 1000000
        self.memory.beta = min(beta_end, self.memory.beta + (beta_end - 0.4) / beta_frames)
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-v5-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250000)
    parser.add_argument("--target-update-frequency", type=int, default=5000)
    parser.add_argument("--replay-start-size", type=int, default=30000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=4)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DDQN-pong-v5", name=args.wandb_run_name, save_code=True, mode="online")
    agent = DQNAgent(args=args)
    agent.run()