import ray, math
from ray import tune
from Model4 import AV20_M4
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=AV20_M4)
result = None
for i in range(1000):
    print(i)
    result = trainer.train()
    if i == 499:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        print(pretty_print(result))
    if i == 999:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        print(pretty_print(result))