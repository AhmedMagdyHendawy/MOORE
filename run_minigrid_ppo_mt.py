# mushroomrl
from mushroom_rl.core import Logger
from mushroom_rl.environments import *
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.parameters import Parameter
# deeplearning
import torch.optim as optim
import torch.nn.functional as F
from moore.core import Core
from moore.algorithms.actor_critic import MTPPO
from moore.policy import MTBoltzmannTorchPolicy
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

MT_EXP = {"MT7": ["MiniGrid-DoorKey-6x6-v0", "MiniGrid-DistShift1-v0",
                           "MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-LavaGapS7-v0",
                           "MiniGrid-MemoryS11-v0", "MiniGrid-SimpleCrossingS9N2-v0", "MiniGrid-MultiRoom-N2-S4-v0"],
          "MT3": ["MiniGrid-LavaGapS7-v0", "MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-MemoryS11-v0"],
          "MT5": ["MiniGrid-DoorKey-6x6-v0", "MiniGrid-LavaGapS7-v0", "MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-DistShift1-v0", "MiniGrid-MemoryS11-v0"]
          }

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
    env_names = MT_EXP[args.env_name]
    horizon = args.horizon
    gamma = args.gamma
    gamma_eval = args.gamma_eval
    
    mdp = []
    for env_name_i in env_names:
        mdp.append(MiniGrid(env_name_i, horizon = horizon, gamma=gamma, render_mode=args.render_mode))

    n_contexts = len(mdp)

    batch_size = args.batch_size
    train_frequency = args.train_frequency

    # Policy
    actor_network = getattr(Network, args.actor_network)#
    actor_n_features = args.actor_n_features#
    lr_actor = args.lr_actor 
    beta=1.
    
    actor_params = dict(beta=beta,
                        n_features=actor_n_features,
                        n_contexts=n_contexts,
                        orthogonal=args.orthogonal,
                        learning_rate = lr_actor,
                        n_experts=args.n_experts,
                        use_cuda=args.use_cuda,
                        )#

    policy = MTBoltzmannTorchPolicy(
            actor_network,
            mdp[0].info.observation_space.shape,
            (mdp[0].info.action_space.n,),
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
                        n_contexts=n_contexts,
                        orthogonal=args.orthogonal,
                        learning_rate = lr_critic,
                        n_experts=args.n_experts,
                        input_shape = mdp[0].info.observation_space.shape,
                        output_shape=(1,))
    
    # alg
    eps = 0.2
    ent_coeff = 0.01
    lam=.95
    alg_params = dict(
            n_epochs_policy=8,
            batch_size=batch_size*n_contexts,
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
    agent = MTPPO(mdp[0].info, policy, n_contexts=n_contexts, **alg_params)

    single_logger.info(agent._V.model.network)
    single_logger.info(agent.policy._logits.model.network)

    os.makedirs(save_dir, exist_ok=True)

    # Algorithm
    core = Core(agent, mdp)

    # # RUN
    # metrics
    metrics = {mdp_i.env_name: {} for mdp_i in mdp}
    for key, value in metrics.items():
        value.update({"MinReturn": []})
        value.update({"MaxReturn": []})
        value.update({"AverageReturn": []})
        value.update({"AverageDiscountedReturn": []})
    metrics.update({"all_minigrid": {"AverageReturn": [], "AverageDiscountedReturn": []}})

    current_all_average_return = 0.0
    current_all_average_discounted_return = 0.0
    for c, mdp_c in enumerate(mdp):
        # random policy evaluation
        core.eval = True
        agent.policy.set_beta(beta)
        core.current_idx = c
        
        dataset = core.evaluate(n_episodes=n_episodes_test, render=args.render_eval)
        min_J, max_J, mean_J, mean_discounted_J, _ = get_stats(dataset, gamma, gamma_eval)
        metrics[mdp_c.env_name]["MinReturn"].append(min_J)
        metrics[mdp_c.env_name]["MaxReturn"].append(max_J)
        metrics[mdp_c.env_name]["AverageReturn"].append(mean_J)
        metrics[mdp_c.env_name]["AverageDiscountedReturn"].append(mean_discounted_J)

        current_all_average_return+=mean_J
        current_all_average_discounted_return+=mean_discounted_J

        single_logger.epoch_info(0,
                            EnvName = mdp_c.env_name,
                            MinReturn=min_J,
                            MaxReturn = max_J,
                            AverageReturn = mean_J,
                            AverageDiscountedReturn = mean_discounted_J,)
        if args.wandb:
            wandb.log({f'{mdp_c.env_name}/MinReturn': min_J,
                        f'{mdp_c.env_name}/MaxReturn': max_J,
                        f'{mdp_c.env_name}/AverageReturn':mean_J,
                        f'{mdp_c.env_name}/AverageDiscountedReturn':mean_discounted_J,
                        }, step = 0, commit=False)
    
    metrics["all_minigrid"]["AverageReturn"].append(current_all_average_return/ n_contexts)
    metrics["all_minigrid"]["AverageDiscountedReturn"].append(current_all_average_discounted_return/ n_contexts)

    if args.wandb:
        wandb.log({"all_minigrid/AverageReturn": current_all_average_return/ n_contexts, 
                   "all_minigrid/AverageDiscountedReturn": current_all_average_discounted_return/ n_contexts}, step = 0, commit=True)
    
    
    for n in trange(n_epochs):
        core.eval = False
        agent.policy.set_beta(beta)
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency, render=args.render_train)

        current_all_average_return = 0.0
        current_all_average_discounted_return = 0.0
        for c, mdp_c in enumerate(mdp):
            core.eval = True
            agent.policy.set_beta(beta)
            core.current_idx = c
            dataset = core.evaluate(n_episodes=n_episodes_test, render=(args.render_eval if n%args.render_interval == 0 and exp_id == 0 else False))
            min_J, max_J, mean_J, mean_discounted_J, _ = get_stats(dataset, gamma, gamma_eval)
            metrics[mdp_c.env_name]["MinReturn"].append(min_J)
            metrics[mdp_c.env_name]["MaxReturn"].append(max_J)
            metrics[mdp_c.env_name]["AverageReturn"].append(mean_J)
            metrics[mdp_c.env_name]["AverageDiscountedReturn"].append(mean_discounted_J)

            current_all_average_return+=mean_J
            current_all_average_discounted_return+=mean_discounted_J

            single_logger.epoch_info(n+1,
                                EnvName = mdp_c.env_name,
                                MinReturn=min_J,
                                MaxReturn = max_J,
                                AverageReturn = mean_J,
                                AverageDiscountedReturn = mean_discounted_J,
                                )
            if args.wandb:
                wandb.log({ f'{mdp_c.env_name}/MinReturn': min_J,
                            f'{mdp_c.env_name}/MaxReturn': max_J,
                            f'{mdp_c.env_name}/AverageReturn':mean_J,
                            f'{mdp_c.env_name}/AverageDiscountedReturn':mean_discounted_J,
                            }, step = n+1, commit=False)  

        metrics["all_minigrid"]["AverageReturn"].append(current_all_average_return/ n_contexts)
        metrics["all_minigrid"]["AverageDiscountedReturn"].append(current_all_average_discounted_return/ n_contexts)

        if args.wandb:
            wandb.log({"all_minigrid/AverageReturn": current_all_average_return/ n_contexts, 
                       "all_minigrid/AverageDiscountedReturn": current_all_average_discounted_return/ n_contexts}, step = n+1, commit=True)


    if args.wandb:
        wandb.finish()

    if "Mixture" in args.actor_network:
        os.makedirs(os.path.join(save_dir, "critic_model"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "actor_model"), exist_ok=True)

        single_logger.info("Saving Shared Backbone and Task Encoder of Critic Network")
        agent._V.model.network.save_shared_backbone(os.path.join(save_dir, "critic_model", "critic_backbone.pth"))
        agent._V.model.network.save_task_encoder(os.path.join(save_dir, "critic_model", "critic_task_encoder.pth"))
        single_logger.info("Saving Shared Backbone and Task Encoder of Actor Network")
        agent.policy._logits.model.network.save_shared_backbone(os.path.join(save_dir, "actor_model", "actor_backbone.pth"))
        agent.policy._logits.model.network.save_task_encoder(os.path.join(save_dir, "actor_model", "actor_task_encoder.pth"))


    return metrics

if __name__ == '__main__':
    # arguments
    args = argparser()

    if args.seed is not None:
        assert len(args.seed) == args.n_exp

    # logging
    results_dir = os.path.join(args.results_dir, "minigrid", "MT", args.env_name)

    logger = Logger(args.exp_name, results_dir=results_dir, log_console=True, use_timestamp=args.use_timestamp)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + MTPPO.__name__)
    logger.info('Experiment Environment: ' + args.env_name)
    logger.info('Experiment Type: ' + "MT-Baseline")
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
            for metric_key in list(value.keys()):
                metric_value = [o[key][metric_key] for o in out]
                np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)              
    else:
        for key, value in out.items():
            for metric_key, metric_value in value.items(): 
                np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)