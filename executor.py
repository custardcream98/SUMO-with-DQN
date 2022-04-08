import ray
from SumoEnv import SumoEnvironment
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print

ray.init()

trainer = dqn.DQNTrainer(env=SumoEnvironment)

episode_reward_mean = 0
while True:
    result = trainer.train()
    
    episode_reward_mean = result['episode_reward_mean']

    # print(f'episodes_total = {result["episodes_total"]}\nepisode_reward_mean = {episode_reward_mean}')
    print(pretty_print(result))
    """
    dict_keys(['episode_reward_max', 'episode_reward_min', 'episode_reward_mean', 'episode_len_mean', 'episode_media', 'episodes_this_iter', 'policy_reward_min', 'policy_reward_max', 'policy_reward_mean', 'custom_metrics', 'hist_stats', 'sampler_perf', 'off_policy_estimator', 'num_healthy_workers', 'timesteps_total', 'timesteps_this_iter', 'agent_timesteps_total', 'timers', 'info', 'done', 'episodes_total', 'training_iteration', 'trial_id', 'experiment_id', 'date', 'timestamp', 'time_this_iter_s', 'time_total_s', 'pid', 'hostname', 'node_ip', 'config', 'time_since_restore', 'timesteps_since_restore', 'iterations_since_restore', 'perf'])
    """