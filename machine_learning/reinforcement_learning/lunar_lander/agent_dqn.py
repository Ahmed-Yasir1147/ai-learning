import gymnasium as gym
import random
import itertools
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dqn import DQN
from experience_replay import ReplayMemory


if torch.cuda.is_available():
    device = "gpu"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):

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

        self.loss = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{param_set}_dqn.log")
        self.RUN_FILE = os.path.join(RUNS_DIR, f"{param_set}_dqn.pt")
        

    def run(self, is_training=False, render=False):
        # Initialization
        env = gym.make("LunarLander-v3", continuous=False)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        online_dqn = DQN(
            input_dim=num_states,
            output_dim=num_actions,
        ).to(device)
        best_reward = float("-inf")

        if is_training:
            target_dqn = DQN(
                input_dim=num_states,
                output_dim=num_actions
            ).to(device)
            target_dqn.load_state_dict(online_dqn.state_dict())
            memory = ReplayMemory(self.replay_memory_size)
            steps = 0
            self.optimizer = optim.Adam(online_dqn.parameters(), lr=self.alpha)

        else:
            online_dqn.load_state_dict(torch.load(self.RUN_FILE))
            online_dqn.eval()

        # Running of model
        for epsiode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            episode_reward = 0
            terminated = False

            # Epsilon greedy
            while not terminated:
                if is_training and random.random() < self.epsilon_init:
                    action = env.action_space.sample() # Explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = online_dqn(state.unsqueeze(0)).squeeze().argmax() # Exploit
                
                next_state, reward, terminated, _, _ = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    memory.add((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                episode_reward += reward.item()

            print(f"Episode: {epsiode} Reward: {episode_reward}")

            if is_training:
                # Epsilon decay
                self.epsilon_init = max(self.epsilon_min, self.epsilon_init * self.epsilon_decay)

                # Training of Online DQN
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, online_dqn, target_dqn)

                # Syncing of target network
                if steps >= self.network_sync_rate:
                    steps = 0
                    target_dqn.load_state_dict(online_dqn.state_dict())

                # Storing best DQN
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    with open(self.LOG_FILE, "a") as file:
                        file.write(f"Episode: {epsiode} Reward: {episode_reward} \n")
                    torch.save(online_dqn.state_dict(), self.RUN_FILE)


    def optimize(self, mini_batch, online_dqn, target_dqn):
        self.optimizer.zero_grad()

        states, actions, next_states, rewards, terminateds = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminateds = torch.tensor(terminateds).float().to(device)

        predict_q = online_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        next_q = target_dqn(next_states).max(dim=1).values
        target_q = rewards + (1 - terminateds) * (self.gamma * next_q)

        loss = self.loss(predict_q, target_q)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("hyperparameters", help="Choose the param set from yaml")
    parser.add_argument("--train", help="Train or test", action="store_true")
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(render=True)

        


        

