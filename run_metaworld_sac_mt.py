import metaworld
# mushroomrl
from mushroom_rl.core import Logger
# deeplearning frameworks
import torch.optim as optim
import torch.nn.functional as F
# continual proto-value functions
from moore.core import VecCore
from moore.algorithms.actor_critic import MTSAC
from moore.environments.metaworld_env import make_env
from moore.environments import SubprocVecEnv
from moore.utils.dataset import get_stats
from moore.utils.argparser import argparser
import moore.utils.networks_sac as Network
# data handling
import numpy as np
# visualization
from tqdm import trange
import wandb
# Utils
import pickle
import os

# The function is used to run a single experiment 
def run_experiment(args, save_dir, exp_id = 0, seed = None):
    import matplotlib
    matplotlib.use('Agg')

    np.random.seed()

    single_logger = Logger(f"seed_{exp_id if seed is None else seed}", results_dir=save_dir, log_console=True)
    save_dir = single_logger.path

    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_episodes_test = args.n_episodes_test

    # MDP
    exp_type = args.exp_type
    # env_name = args.env_name
    horizon = args.horizon
    gamma = args.gamma
    gamma_eval = args.gamma_eval

    benchmark = getattr(metaworld, exp_type)()

    mdp = SubprocVecEnv(
        [make_env(env_name=env_name, 
                  env_cls=env_cls, 
                  train_tasks = benchmark.train_tasks, 
                  horizon=horizon, 
                  gamma=gamma, 
                  normalize_reward=args.normalize_reward,
                  sample_task_per_episode=args.sample_task_per_episode)
          for env_name, env_cls in benchmark.train_classes.items()])
    
    n_contexts = mdp.num_envs

    # Settings
    initial_replay_size = args.initial_replay_size #
    max_replay_size = int(args.max_replay_size) #
    batch_size = args.batch_size #
    train_frequency = args.train_frequency
    tau = args.tau #
    warmup_transitions = args.warmup_transitions #
    log_std_min = args.log_std_min
    log_std_max = args.log_std_max

    append_context_actor = "Single" in args.actor_network
    append_context_mu_actor = "Single" in args.actor_mu_network
    append_context_sigma_actor = "Single" in args.actor_sigma_network
    append_context_critic = "Single" in args.critic_network

    # Settings
    if args.shared_mu_sigma:
        actor_network = getattr(Network, args.actor_network)#
        actor_n_features = args.actor_n_features#
    else:
        actor_mu_network = getattr(Network, args.actor_mu_network)#
        actor_sigma_network = getattr(Network, args.actor_sigma_network) #
        actor_mu_n_features = args.actor_mu_n_features#
        actor_sigma_n_features = args.actor_sigma_n_features#

    critic_network = getattr(Network, args.critic_network)#
    critic_n_features = args.critic_n_features#

    lr_alpha = args.lr_alpha #
    lr_actor = args.lr_actor #
    lr_critic = args.lr_critic #

    target_entropy = args.target_entropy #
    
    actor_params = None
    actor_mu_params = None
    actor_sigma_params = None

    # Approximator
    if args.shared_mu_sigma:
        actor_input_shape = mdp.observation_space.shape
        if append_context_actor:
            single_logger.info("Append context to Actor's input!")
            actor_input_shape = (actor_input_shape[0] + n_contexts,)

        actor_params = dict(network=actor_network,
                            n_features=actor_n_features,
                            input_shape=actor_input_shape,
                            output_shape=mdp.action_space.shape,
                            shared_mu_sigma = args.shared_mu_sigma,
                            use_cuda=args.use_cuda,
                            n_contexts = n_contexts,
                            activation = args.activation,
                            orthogonal = args.orthogonal,
                            n_experts = args.n_experts,
                            agg_activation = args.agg_activation,
                            )#
    else:
        mu_actor_input_shape = mdp.observation_space.shape
        if append_context_mu_actor:
            single_logger.info("Append context to Mu Actor's input!")
            mu_actor_input_shape = (mu_actor_input_shape[0] + n_contexts,)
        actor_mu_params = dict(network=actor_mu_network,
                                n_features=actor_mu_n_features,
                                input_shape=mu_actor_input_shape,
                                output_shape=mdp.action_space.shape,
                                use_cuda=args.use_cuda,
                                n_contexts = n_contexts,
                                activation = args.activation,
                                orthogonal = args.orthogonal,
                                n_experts = args.n_experts,
                                agg_activation = args.agg_activation,)#

        sigma_actor_input_shape = mdp.observation_space.shape
        if append_context_sigma_actor:
            single_logger.info("Append context to Sigma Actor's input!")
            sigma_actor_input_shape = (sigma_actor_input_shape[0] + n_contexts,)
        actor_sigma_params = dict(network=actor_sigma_network,
                                    n_features=actor_sigma_n_features,
                                    input_shape=sigma_actor_input_shape,
                                    output_shape=mdp.action_space.shape,
                                    use_cuda=args.use_cuda,
                                    n_contexts = n_contexts,
                                    activation = args.activation,
                                    orthogonal = args.orthogonal,
                                    n_experts = args.n_experts,
                                    agg_activation = args.agg_activation,)#

    actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': lr_actor, 'betas': (0.9, 0.999)}}#

    critic_input_shape = (mdp.observation_space.shape[0] + mdp.action_space.shape[0],)
    if append_context_critic:
        single_logger.info("Append context to Critic's input!")
        critic_input_shape = (critic_input_shape[0]+n_contexts,)

    critic_params = dict(network=critic_network,
                            optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic, 'betas': (0.9, 0.999)}},
                            loss=F.mse_loss,
                            n_features=critic_n_features,
                            input_shape=critic_input_shape,
                            output_shape=(1,),
                            use_cuda=args.use_cuda,
                            n_contexts = n_contexts,
                            activation = args.activation,
                            orthogonal = args.orthogonal,
                            n_experts = args.n_experts,
                            agg_activation = args.agg_activation,
                            )

    if args.debug:
        initial_replay_size = 150
        batch_size = 8
        n_epochs = 2
        n_steps = 150
        n_steps_test = 100
        n_episodes_test = 1
        args.wandb = False
        warmup_transitions = 150
    
    if args.wandb:
        wandb.init(name = "seed_"+str(exp_id if seed is None else seed), project = "MOORE", group = f"metaworld_{args.env_name}" if args.env_name is not None else f"metaworld_{args.exp_type}", job_type=args.exp_name, entity=args.wandb_entity, config=vars(args))

    # create SAC agent
    agent = MTSAC(mdp_info=mdp.info,
                    batch_size=batch_size, initial_replay_size=initial_replay_size,
                    max_replay_size=max_replay_size,
                    warmup_transitions=warmup_transitions, tau=tau, lr_alpha=lr_alpha,
                    actor_params = actor_params, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                    actor_optimizer=actor_optimizer, critic_params=critic_params,
                    target_entropy=target_entropy, critic_fit_params=None,
                    log_std_min=log_std_min, log_std_max=log_std_max, shared_mu_sigma=args.shared_mu_sigma,
                    n_contexts=n_contexts)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "actor"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "critic"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "agent"), exist_ok=True)
    
    # load agent
    if args.load_agent:
        agent = agent.load(args.load_agent)
    else:
        # load the critic
        if args.load_critic:
            agent.set_critic_weights(args.load_critic)
        # load the policy/ actor
        if args.load_actor:
            agent.policy.set_weights(np.load(args.load_actor))

    # set logger
    agent.set_logger(single_logger)
    # log models summary
    agent.models_summary()

    # Algorithm
    core = VecCore(agent, mdp)
    
    # metrics
    env_names = mdp.get_attr("env_name")
    
    metrics = {env_name_i: {} for env_name_i in env_names}
    for key, value in metrics.items():
        value.update({"MinReturn": []})
        value.update({"MaxReturn": []})
        value.update({"AverageReturn": []})
        value.update({"AverageDiscountedReturn": []})
        value.update({"SuccessRate": []})
        value.update({"LogAlpha": []})
    metrics.update({"all_metaworld": {"SuccessRate": []}})

    if args.start_epoch == 0:
        # Intialize the replay memory
        core.eval = False
        core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, render=args.render_train)

        # random policy evaluation
        current_success_rate_avg = 0.0
        for c, key in enumerate(env_names):
            core.eval = True
            core.current_idx = c
            dataset, dataset_info = core.evaluate(n_episodes=n_episodes_test, render=args.render_eval if exp_id == 0 else False, get_env_info=True)
            min_J, max_J, mean_J, mean_discounted_J, success_rate = get_stats(dataset, gamma, gamma_eval, dataset_info=dataset_info)
            
            log_alpha = agent.get_log_alpha(c)
            
            metrics[key]["MinReturn"].append(min_J)
            metrics[key]["MaxReturn"].append(max_J)
            metrics[key]["AverageReturn"].append(mean_J)
            metrics[key]["AverageDiscountedReturn"].append(mean_discounted_J)
            metrics[key]["SuccessRate"].append(success_rate)
            metrics[key]["LogAlpha"].append(log_alpha) 

            current_success_rate_avg+=success_rate

            single_logger.epoch_info(0, C = key,
                                min_J=min_J,
                                max_J = max_J,
                                mean_J = mean_J,
                                mean_discounted_J = mean_discounted_J,
                                success_rate = success_rate,
                                log_alpha = log_alpha)
            if args.wandb:
                wandb.log({f'{key}/MinReturn': min_J,
                            f'{key}/MaxReturn': max_J,
                            f'{key}/AverageReturn':mean_J,
                            f'{key}/AverageDiscountedReturn':mean_discounted_J,
                            f'{key}/SuccessRate':success_rate,
                            f'{key}/LogAlpha':log_alpha}, step = 0, commit=False)

        metrics["all_metaworld"]["SuccessRate"].append(current_success_rate_avg / n_contexts)

        if args.wandb:
            wandb.log({"all_metaworld/SuccessRate": current_success_rate_avg / n_contexts}, step = 0, commit=True)

    for n in trange(args.start_epoch, n_epochs):
        
        # train
        core.eval = False
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency, render=args.render_train, quiet=False)
        
        # eval
        core.eval = True
        current_success_rate_avg = 0
        for c, key in enumerate(env_names):
            core.current_idx = c
            dataset, dataset_info = core.evaluate(n_episodes=n_episodes_test, render=(args.render_eval if n%args.render_interval == 0 and exp_id == 0 else False), get_env_info=True)
            min_J, max_J, mean_J, mean_discounted_J, success_rate = get_stats(dataset, gamma, gamma_eval, dataset_info=dataset_info)
            
            log_alpha = agent.get_log_alpha(c)

            metrics[key]["MinReturn"].append(min_J)
            metrics[key]["MaxReturn"].append(max_J)
            metrics[key]["AverageReturn"].append(mean_J)
            metrics[key]["AverageDiscountedReturn"].append(mean_discounted_J)
            metrics[key]["SuccessRate"].append(success_rate)
            metrics[key]["LogAlpha"].append(log_alpha) 

            current_success_rate_avg+=success_rate

            single_logger.epoch_info(n+1, C = key,
                                min_J=min_J,
                                max_J = max_J,
                                mean_J = mean_J,
                                mean_discounted_J = mean_discounted_J,
                                success_rate = success_rate,
                                log_alpha = log_alpha)
            if args.wandb:
                wandb.log({f'{key}/MinReturn': min_J,
                            f'{key}/MaxReturn': max_J,
                            f'{key}/AverageReturn':mean_J,
                            f'{key}/AverageDiscountedReturn':mean_discounted_J,
                            f'{key}/SuccessRate':success_rate,
                            f'{key}/LogAlpha':log_alpha}, step = n+1, commit=False)

        metrics["all_metaworld"]["SuccessRate"].append(current_success_rate_avg / n_contexts)

        if args.wandb:
            wandb.log({"all_metaworld/SuccessRate": current_success_rate_avg / n_contexts}, step = n+1, commit=True)

        if (n+1) % args.rl_checkpoint_interval == 0:
            # save the learned policy/ actor so far
            actor_weights = agent.policy.get_weights()
            np.save(os.path.join(save_dir, f"actor/actor_weights_{n+1}.npy"), actor_weights)
            # save the critic so far
            critic_weights = agent.get_critic_weights()
            for key, value in critic_weights.items():
                np.save(os.path.join(save_dir, f"critic/{key}_{n+1}.npy"), value)
            # save the whole agent
            agent.save(os.path.join(save_dir, f"agent/agent_{n+1}"), full_save=True)


    if args.wandb:
        wandb.finish()

    # save the learned policy/ actor
    actor_weights = agent.policy.get_weights()
    np.save(os.path.join(save_dir, "actor/actor_weights.npy"), actor_weights)
    # save the critic
    critic_weights = agent.get_critic_weights()
    for key, value in critic_weights.items():
        np.save(os.path.join(save_dir, f"critic/{key}.npy"), value)
    # save the whole agent
    agent.save(os.path.join(save_dir, f"agent/agent_final"), full_save=True)

    return metrics

if __name__ == '__main__':
    # arguments
    args = argparser()

    if args.seed is not None:
        assert len(args.seed) == args.n_exp

    alg_name = "mixture_orthogonal_experts" if args.orthogonal else "mixture_experts"
    # logging
    results_dir = os.path.join(args.results_dir, args.exp_type, alg_name)

    logger = Logger(args.exp_name, results_dir=results_dir, log_console=True, use_timestamp=args.use_timestamp)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + MTSAC.__name__)

    save_dir = logger.path

    with open(os.path.join(save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    
    logger.info(vars(args))

    out = run_experiment(args, save_dir, seed=args.seed)
    
    for key, value in out.items():
        if key == "all_metaworld":
            np.save(os.path.join(save_dir, f'all_SuccessRate.npy'), value["SuccessRate"])
        else:
            for metric_key, metric_value in value.items(): 
                np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)
                
                