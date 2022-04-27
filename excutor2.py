import ray, math
from ray import tune
from Model5_S import AV100_M5_S
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=AV100_M5_S)
result = None

trainer.restore(r'C:\Users\User/ray_results\DQNTrainer_AV100_M5_S_2022-04-25_15-08-41iexf1lvp\checkpoint_001000\checkpoint-1000')

for i in range(1000):
    result = trainer.train()
checkpoint = trainer.save()
print("checkpoint saved at", checkpoint)
print(pretty_print(result))