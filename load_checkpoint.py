import ray
from SumoEnv import SumoEnvironment
from ray.rllib.agents import dqn


ray.init()

config = {
    "env_config": {
        "is_test_run": True,
    }
}

trainer = dqn.DQNTrainer(config=config, env=SumoEnvironment)

# 먼저 체크포인트를 불러온다
trainer.restore(r'C:\Users\User/ray_results\DQNTrainer_SumoEnvironment_2022-04-11_16-33-2529_pko2n\checkpoint_000009\checkpoint-9')

# 테스트 Run
trainer.train()