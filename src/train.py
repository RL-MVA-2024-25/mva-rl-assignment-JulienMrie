from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env



env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.model = None
        self.log = True
        self.model_name = "dqn_hiv"
        vec_env = make_vec_env(HIVPatient, n_envs=1)
        self.env = VecNormalize.load("vec_normalize.pkl", vec_env)
        self.mean = self.env.obs_rms.mean
        self.var = self.env.obs_rms.var
        self.epsilon = 1e-8
        self.clip_obs = 10.0
    
    def act(self, observation, use_random=False):
        if(self.model == None):
            return 0
        observation = np.clip((observation - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

        action, _states = self.model.predict(observation, deterministic = False)
      
        return action        

    def save(self, path):
        pass

    def load(self):
        print(self.mean, self.var)
        print("Loading model:", self.model_name)
        self.model = DQN.load(self.model_name)