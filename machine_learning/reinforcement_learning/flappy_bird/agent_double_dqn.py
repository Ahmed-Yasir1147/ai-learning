
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
import random
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

# Detecting device to make use of best processor available
if torch.backends.mps.is_available(): # Apple GPU
    device="mps"
elif torch.cuda.is_available(): # Nvidia GPU
    device = "cuda"
else: # no GPU
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as file:
            all_param_set = yaml.safe_load(file)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}double_dqn.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}double_dqn.pt")


    
    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(
            num_states, # input_dim
            num_actions, # output_dim
        ).to(device)

        if is_training:
            memory  = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = float("-inf")
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            episode_reward = 0
            terminated = False

            state = torch.tensor(state, dtype=torch.float32, device=device) # neural network can only process tensor

            while not terminated and episode_reward < self.reward_threshold:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit

                next_state, reward, terminated, _, _ = env.step(action) # agent taking action

                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)


                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1
                
                episode_reward += reward.item()

                state = next_state 



            if is_training:
                print(f"Episode: {episode} Epsilon: {epsilon} Reward: {episode_reward}")
            else:
                print(f"Reward: {episode_reward}")

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # training after getting certain memory points to avoid correlated points problem
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                # update target network after certain steps
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

                if episode_reward > best_reward:
                    best_reward = episode_reward

                    with open(self.LOG_FILE, "a") as f:
                        f.write(f"Best reward: {best_reward} Episode: {episode}\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)


        # env.close()
    

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, next_state, reward, terminated in mini_batch:
            if terminated:
                target_q = reward

            else:
                with torch.no_grad():
                    next_action = policy_dqn(next_state).argmax() # policy network will decide action
                    next_state_q = target_dqn(next_state)[next_action] # target network will evaluate q value for action decided by policy dqn
                    target_q = reward + self.gamma * next_state_q # y_actual

            current_q = policy_dqn(state) # y_predicted

            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q, target_q)
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="training_mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)