import ray, math
from ray import tune
from Final import SumoEnvironment
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=SumoEnvironment)
# trainer = dqn.DQNTrainer(env=SumoEnvironment)

episode_reward_mean = 0

for i in range(30):
    result = trainer.train()

    # if i % 4 == 0 and i != 0:
    #     episode_reward_mean = result['episode_reward_mean'] if not math.isnan(result['episode_reward_mean']) else 0
    #     print(f'episodes_total = {result["episodes_total"]}\nsum of episode_reward_mean = {episode_reward_mean}')
    #     checkpoint = trainer.save()
        # print("checkpoint saved at", checkpoint)
    # print(pretty_print(result))

   