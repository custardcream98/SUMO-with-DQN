import ray
from ActionSetting import SumoEnvironment
from ray.rllib.agents import dqn

ray.init()

trainer = dqn.DQNTrainer(env=SumoEnvironment)

while True:
    result = trainer.train()
