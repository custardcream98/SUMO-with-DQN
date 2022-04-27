import ray, math
from ray import tune
from Model5_Q import AV100_M5_Q
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=AV100_M5_Q)
result = None
for i in range(1000):
    result = trainer.train()
checkpoint = trainer.save()
print("checkpoint saved at", checkpoint)
print(pretty_print(result))