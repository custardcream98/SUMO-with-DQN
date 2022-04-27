import ray, math
from ray import tune
from Model1_ql2 import SumoEnvironment
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=SumoEnvironment)

while True:
    result = trainer.train()
   