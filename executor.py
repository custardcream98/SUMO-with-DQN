import ray
from SumoEnv import SumoEnvironment
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=SumoEnvironment)

while True:
    result = trainer.train()
    print(type(result))
    print(result.keys())
    #print(pretty_print(result))