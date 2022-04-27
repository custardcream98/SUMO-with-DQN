import ray, math
from ray import tune
from SumoEnv import SumoEnvironment
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

config = {
    "rollout_fragment_length": 300,
    "batch_mode": "truncate_episodes",
    # "batch_size": 450,
    # "batch_mode": "truncate_episodes",
    # "min_sample_timesteps_per_reporting": 3600
    "train_batch_size": 600,
    "env_config": {
        "is_test_run": False,
    }
}

trainer = dqn.DQNTrainer(config=config, env=SumoEnvironment)
# trainer = dqn.DQNTrainer(env=SumoEnvironment)

episode_reward_mean = 0

for i in range(10):
    result = trainer.train()

    if i % 4 == 0 and i != 0:
        episode_reward_mean = result['episode_reward_mean'] if not math.isnan(result['episode_reward_mean']) else 0
        print(f'episodes_total = {result["episodes_total"]}\nsum of episode_reward_mean = {episode_reward_mean}')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
    print(pretty_print(result))

   