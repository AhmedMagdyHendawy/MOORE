# mushroomrl
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.policy import BoltzmannTorchPolicy
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.parameters import Parameter
# deeplearning
import torch.optim as optim
import torch.nn.functional as F
# continual proto-value functions 
from moore.environments import MiniGrid
from moore.utils.argparser import argparser
from moore.utils.dataset import get_stats
import moore.utils.networks_ppo as Network
# visulization
from tqdm import trange
import wandb
# Utils
import os
import pickle
from joblib import delayed, Parallel


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
    env_name = args.env_name
    horizon = args.horizon
    gamma = args.gamma
    gamma_eval = args.gamma_eval

    mdp = MiniGrid(env_name, horizon = horizon, gamma=gamma, render_mode=args.render_mode)

    batch_size = args.batch_size
    train_frequency = args.train_frequency

    # Policy
    actor_network = getattr(Network, args.actor_network)#
    actor_n_features = args.actor_n_features#
    lr_actor = args.lr_actor #
    beta=1.
    
    actor_params = dict(beta=beta,
                        n_features=actor_n_features,
                        use_cuda=args.use_cuda,
                        )#

    policy = BoltzmannTorchPolicy(
            actor_network,
            mdp.info.observation_space.shape,
            (mdp.info.action_space.n,),
            **actor_params)
    
    actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': lr_actor, 'betas': (0.9, 0.999)}}#
    
    # critic
    critic_network = getattr(Network, args.critic_network)#
    critic_n_features = args.critic_n_features#
    lr_critic = args.lr_critic #
    critic_fit_params = None
    critic_params = dict(
                        network=critic_network,
                        optimizer={
                            'class': optim.Adam, 
                            'params': {'lr': lr_critic, 'betas': (0.9, 0.999)}},
                        loss=F.mse_loss,
                        n_features=critic_n_features,
                        batch_size=batch_size,
                        input_shape = mdp.info.observation_space.shape,
                        output_shape=(1,))
    
    # alg
    eps = 0.2
    ent_coeff = 0.01
    lam=.95
    alg_params = dict(
            n_epochs_policy=8,
            batch_size=batch_size,
            eps_ppo=eps,
            ent_coeff=ent_coeff,
            lam=lam,
            actor_optimizer = actor_optimizer,
            critic_params = critic_params,
            critic_fit_params=critic_fit_params)

    if args.debug:
        batch_size = 8
        n_epochs = 2
        n_steps = 150
        n_steps_test = 100
        n_episodes_test = 1
        args.wandb = False

    if args.wandb:
        wandb.init(name = "seed_"+str(exp_id if seed is None else seed), project = "MOORE", group = f"minigrid_{args.env_name}", job_type=args.exp_name, entity=args.wandb_entity, config=vars(args))

    # Agent
    agent = PPO(mdp.info, policy, **alg_params)

    single_logger.info(agent._V.model.network)
    single_logger.info(agent.policy._logits.model.network)
    
    os.makedirs(save_dir, exist_ok=True)

    # Algorithm
    core = Core(agent, mdp)

    # # RUN
    # metrics
    metrics = {mdp_i.env_name: {} for mdp_i in [mdp]}
    for key, value in metrics.items():
        value.update({"MinReturn": []})
        value.update({"MaxReturn": []})
        value.update({"AverageReturn": []})
        value.update({"AverageDiscountedReturn": []})

    # random policy evaluation
    agent.policy.set_beta(beta)
    mdp.eval = True
    dataset = core.evaluate(n_episodes=n_episodes_test, render=args.render_eval)
    min_J, max_J, mean_J, mean_discounted_J, success_rate = get_stats(dataset, gamma, gamma_eval)
    metrics[mdp.env_name]["MinReturn"].append(min_J)
    metrics[mdp.env_name]["MaxReturn"].append(max_J)
    metrics[mdp.env_name]["AverageReturn"].append(mean_J)
    metrics[mdp.env_name]["AverageDiscountedReturn"].append(mean_discounted_J)

    single_logger.epoch_info(0,
                        MinReturn=min_J,
                        MaxReturn = max_J,
                        AverageReturn = mean_J,
                        AverageDiscountedReturn = mean_discounted_J,)
    if args.wandb:
        wandb.log({f'{mdp.env_name}/MinReturn': min_J,
                    f'{mdp.env_name}/MaxReturn': max_J,
                    f'{mdp.env_name}/AverageReturn':mean_J,
                    f'{mdp.env_name}/AverageDiscountedReturn':mean_discounted_J,
                    }, step = 0, commit=False)
    
    for n in trange(n_epochs):
        agent.policy.set_beta(beta)
        mdp.eval = False
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency, render=args.render_train)

        agent.policy.set_beta(beta)
        mdp.eval = True
        dataset = core.evaluate(n_episodes=n_episodes_test, render=(args.render_eval if n%args.render_interval == 0 and exp_id == 0 else False))
        min_J, max_J, mean_J, mean_discounted_J, success_rate = get_stats(dataset, gamma, gamma_eval)
        metrics[mdp.env_name]["MinReturn"].append(min_J)
        metrics[mdp.env_name]["MaxReturn"].append(max_J)
        metrics[mdp.env_name]["AverageReturn"].append(mean_J)
        metrics[mdp.env_name]["AverageDiscountedReturn"].append(mean_discounted_J)


        single_logger.epoch_info(n+1,
                            MinReturn=min_J,
                            MaxReturn = max_J,
                            AverageReturn = mean_J,
                            AverageDiscountedReturn = mean_discounted_J,
                            )
        if args.wandb:
            wandb.log({ f'{mdp.env_name}/MinReturn': min_J,
                        f'{mdp.env_name}/MaxReturn': max_J,
                        f'{mdp.env_name}/AverageReturn':mean_J,
                        f'{mdp.env_name}/AverageDiscountedReturn':mean_discounted_J,
                        }, step = n+1, commit=False)  

    if args.wandb:
        wandb.finish()

    if exp_id == 0: 
        core.evaluate(n_episodes=n_episodes_test, render=args.render_final)

    return metrics

if __name__ == '__main__':
    # arguments
    args = argparser()

    if args.seed is not None:
        assert len(args.seed) == args.n_exp

    # logging
    results_dir = os.path.join(args.results_dir, "minigrid", "ST", args.env_name)

    logger = Logger(args.exp_name, results_dir=results_dir, log_console=True, use_timestamp=args.use_timestamp)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + PPO.__name__)
    logger.info('Experiment Environment: ' + args.env_name)
    logger.info('Experiment Type: ' + "Baseline")
    logger.info("Experiment Name: " + args.exp_name)

    save_dir = logger.path

    with open(os.path.join(save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    if args.seed is not None:
        out = Parallel(n_jobs=-1)(delayed(run_experiment)(args, save_dir, i, s)
                              for i, s in zip(range(args.n_exp), args.seed))
    elif args.n_exp > 1:
        out = Parallel(n_jobs=-1)(delayed(run_experiment)(args, save_dir, i)
                              for i in range(args.n_exp))
    else:
        out = run_experiment(args, save_dir)
    
    if args.n_exp > 1:
        for key, value in out[0].items():
            if key == "all":
                all_SuccessRate = [o["all"]["SuccessRate"] for o in out]
                np.save(os.path.join(save_dir, f'all_SuccessRate.npy'), np.vstack(all_SuccessRate))
            else:
                for metric_key in list(value.keys()):
                    metric_value = [o[key][metric_key] for o in out]
                    np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)              
    else:
        for key, value in out.items():
            if key == "all":
                np.save(os.path.join(save_dir, f'all_SuccessRate.npy'), value["SuccessRate"])
            else:
                for metric_key, metric_value in value.items(): 
                    np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)