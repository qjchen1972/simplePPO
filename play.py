import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import gzip 
import argparse
import retro        

from ppo.mpenv import SubprocVecEnv
import ppo.model as md
import ppo.utils as utils
import ppo.buffer as buffer
from inter import savestate 
from street_fighter_custom_wrapper import StreetFighterCustomWrapper

def make_env(game, state, seed=0, rendering=False):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE,
            #players=2            
        )
        #print("buttons:" + str(env.buttons))
        env = StreetFighterCustomWrapper(env, rendering=rendering)
        env.seed(seed)
        return env
    return _init
    
class Trainer:
    def __init__(self, env, resume=None):
        self.env = env
        self.device = utils.get_device('auto')
        self.model = md.PPO([3,100,128]).to(self.device)
        self.n_rollout_steps = 512
        self.n_epochs = 4
        self.batch_size=512
        self.gamma=0.94
        self.clip_range = 0.2
        self.total_timesteps = 100000
        self.lr_schedule = utils.linear_schedule(2.5e-4, 2.5e-6)
        self.clip_range_schedule = utils.linear_schedule(0.15, 0.025)
        
        self.roll = buffer.RolloutBuffer(self.n_rollout_steps, 
                             self.env.observation_space, 
                             self.env.action_space,
                             device='auto', 
                             gamma=self.gamma, 
                             gae_lambda=0.95, 
                             n_envs=self.env.num_envs)
        self.optimizer = self.model.configure_optimizers(lr=self.lr_schedule(1))
        self.last_obs = self.env.reset()
        self.last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        if resume is not None:
            self.model, self.num_timesteps = self.load(self.model, resume, 
                                                self.device)
        else:
            self.num_timesteps = 0        
            
       
    def save(self, num_timesteps, modelpath='./models'):
        ckpt_path = os.path.join(modelpath, f'model_{num_timesteps}.pt')
        raw_model = self.model.module if hasattr(self.model, "module") else self.model      
        checkpoint = {
                    'model': raw_model.state_dict(),
                    'num_timesteps': num_timesteps
                    }
        print(f"saving checkpoint to {ckpt_path}")
        torch.save(checkpoint, ckpt_path)
        
    @classmethod    
    def load(cls, model, num, device, modelpath='./models'):
        ckpt_path = os.path.join(modelpath, f'model_{num}.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)        
        num_timesteps = checkpoint['num_timesteps']
        model.load_state_dict(checkpoint['model']) 
        return model, num_timesteps        
               
        
    def collect_rollouts(self, rendering=False):
        n_steps = 0
        self.roll.reset()
        score = []
        self.model.train(False)
        
        while n_steps < self.n_rollout_steps:
            with torch.no_grad():
                obs_tensor = utils.obs_as_tensor(self.last_obs, self.device)
                actions, values, log_probs = self.model(obs_tensor)
            actions = actions.cpu().numpy()            
            new_obs, rewards, dones, infos = self.env.step(actions)            
            score.append(rewards)
            if rendering: self.env.render()
            n_steps += 1
            self.roll.add(self.last_obs, actions, rewards, 
                          self.last_episode_starts, values, log_probs)
            
            '''
            for idx, done in enumerate(dones):
                if (done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = utils.obs_as_tensor(infos[idx]["terminal_observation"][None],
                              device)                 
                    with torch.no_grad():
                        terminal_value = model.predict_values(terminal_obs)[0]
                    rewards[idx] += gamma * terminal_value
            '''
            
            self.last_obs = new_obs
            self.last_episode_starts = dones
        with torch.no_grad():
            values = self.model.predict_values(utils.obs_as_tensor(new_obs, 
                                              self.device))           
        self.roll.compute_returns_and_advantage(last_values=values, dones=dones)
        score = np.array(score)
        return True, score.sum(axis=0), score.sum()

    def train(self):
        self.model.train(True)
        progress = 1.0 - float(self.num_timesteps) / float(self.total_timesteps)
        lr = self.lr_schedule(progress)    
        if self.clip_range_schedule is not None:        
            clip_range_vf = self.clip_range_schedule(progress)              
        
        utils.update_learning_rate(self.optimizer, lr)
        for epoch in range(self.n_epochs):    
            for rollout_data in self.roll.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.model.evaluate_actions(
                                               rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                                            advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range,
                                         1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                if self.clip_range_schedule is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )                    
                values_pred = values
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                #
                ent_coef = 0.0
                vf_coef = 0.5
                loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
    def learn(self):
        total_score = 0.0
        while self.num_timesteps < self.total_timesteps:        
            continue_training, _, score = self.collect_rollouts(rendering=False)
            if continue_training is False:
                break
            self.num_timesteps += self.env.num_envs    
            self.train()
            total_score += score
            if self.num_timesteps % (10 * self.env.num_envs) == 0:
                self.save(self.num_timesteps)
                print(f'score is {total_score:.4f}')
                total_score = 0.0
        self.env.close()        
              
        
def train(resume=None):

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level12.RyuVsBison"
    NUM_ENV = 16
    env = SubprocVecEnv([make_env(game, state=state, seed=i) for i in range(NUM_ENV)])  
    proc = Trainer(env, resume=resume)
    proc.learn()
    
            
def test(modelNo):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state = "Champion.Level12.RyuVsBison"
    #state = "Champion.Level1.RyuVsGuile"
    #state = "Champion.Level12.RyuVsBison.2Player_v0425"
    env = make_env(game, state=state, rendering=True)()    
    RANDOM_ACTION = False
    device = utils.get_device('cpu')
    if  not RANDOM_ACTION:
        model = md.PPO([3,100,128]).to(device)
        model, _ = Trainer.load(model, modelNo, device)
        model.train(False)
    
    obs = env.reset()
    done = False

    num_episodes = 20
    episode_reward_sum = 0
    num_victory = 0    

    print("\nFighting Begins!\n")
    
    for _ in range(num_episodes):
        done = False
        obs = env.reset()
        
        total_reward = 0

        while not done:
            timestamp = time.time()

            if RANDOM_ACTION:
                obs, reward, done, info = env.step(env.action_space.sample())
                #print(env.action_space.sample())
            else:
                action = model.predict(utils.obs_as_tensor(obs, device)[None])
                action = action.squeeze(dim=0).cpu().numpy()
                #action = np.concatenate((action, playact))
                obs, reward, done, info = env.step(action)
            env.render()
            
            if reward != 0:
                total_reward += reward
                print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
            
            if info['enemy_hp'] < 0 or info['agent_hp'] < 0:
                done = True

        if info['enemy_hp'] < 0:
            print("Victory!")
            num_victory += 1
        print("Total reward: {}\n".format(total_reward))
        episode_reward_sum += total_reward
    

    env.close()
    print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
    if RANDOM_ACTION:
        print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
    else:
        print("Average reward for {}: {}".format(num, episode_reward_sum/num_episodes))
    
if __name__ == '__main__':   
 
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 0)')  
    args = parser.parse_args()
    
    if args.m == 0:
        resume = None
        train(resume=resume)
        print('train...')        
    elif args.m == 1:
        modelNo = 7200
        test(modelNo)
    elif args.m == 2:
        retro_directory = os.path.dirname(retro.__file__)
        game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"
        print(os.path.join(retro_directory, game_dir))
    elif args.m == 3:
        #修改retro目录下的/example/interactivate.py
        #用于保存state,可以2人对战
        path="data/1player.state"
        savestate(path=path)
        
        
        
        
        