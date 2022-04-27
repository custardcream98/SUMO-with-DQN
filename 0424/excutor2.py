import ray, math
from ray import tune
from Model2 import SumoEnvironment
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=SumoEnvironment)
result = None
for i in range(500):
    result = trainer.train()
checkpoint = trainer.save()
print("checkpoint saved at", checkpoint)
print(pretty_print(result))

   