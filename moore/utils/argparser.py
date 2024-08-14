import argparse

def argparser():
     # Argument parser
     parser = argparse.ArgumentParser()

     arg_mdp = parser.add_argument_group('mdp')
     arg_mdp.add_argument('--env_name', type=str, help='Name of the environment.')#
     arg_mdp.add_argument("--horizon", type=int)
     arg_mdp.add_argument("--gamma", type=float, default=0.98)
     arg_mdp.add_argument("--gamma_eval", type=float, default=1.)

     arg_alg = parser.add_argument_group('Algorithm')
     arg_alg.add_argument("--train_frequency", type=int, default=1)
     arg_alg.add_argument("--batch_size", type=int, default=128,
                          help='Batch size for each fit of the network.')
     arg_alg.add_argument("--n_epochs", type=int, default=200,
                          help='Number of epochs.')
     arg_alg.add_argument("--n_steps", type=int,
                          help='Number of learning steps per epoch.')
     arg_alg.add_argument("--n_episodes", type=int,
                          help='Number of learning episodes per epoch.')
     arg_alg.add_argument("--n_steps_test", type=int,
                          help='Number of learning steps per epoch.')
     arg_alg.add_argument("--n_episodes_test", type=int,
                          help='Number of episodes (rollouts) for evaluation per epoch.')
     arg_alg.add_argument("--load_critic", type=str, default=None,
                          help='Directory of the weights for critic')
     arg_alg.add_argument("--load_actor", type=str, default=None,
                          help='Directory of the weights for actor')
     arg_alg.add_argument("--load_agent", type=str, default=None,
                          help='Directory of the saved agent')
     arg_alg.add_argument('--dropout', action='store_true',
                            help='Use dropout in the network')#
     
     arg_alg.add_argument("--actor_network", type=str, default="ActorNetwork")
     arg_alg.add_argument("--actor_n_features", type=int, nargs='+', default=[])
     arg_alg.add_argument("--critic_network", type=str, default="CriticNetwork")
     arg_alg.add_argument("--critic_n_features", type=int, nargs='+', default=[])
     arg_alg.add_argument("--lr_actor", type=float, default=3e-4)
     arg_alg.add_argument("--lr_critic", type=float, default=3e-4)


     arg_me = parser.add_argument_group('MixtureExperts')
     arg_me.add_argument("--orthogonal", action="store_true")
     arg_me.add_argument("--n_experts", type=int, default=4)

     arg_utils = parser.add_argument_group('Utils')
     arg_utils.add_argument('--use_cuda', action='store_true',
                            help='Flag specifying whether to use the GPU.')#
     arg_utils.add_argument('--render_train', action='store_true',
                            help='Flag specifying whether to render the training.')#
     arg_utils.add_argument('--render_eval', action='store_true',
                            help='Flag specifying whether to render the evaluation.')#
     arg_utils.add_argument('--render_final', action='store_true',
                            help='Flag specifying whether to render the evaluation of final policy.')#
     arg_utils.add_argument('--render_interval', type=int, default= 1,
                            help='Render the environment every n epochs')#
     arg_utils.add_argument("--render_mode", type=str, default="rgb_array")
     arg_utils.add_argument("--rl_checkpoint_interval", type=int)
     arg_utils.add_argument('--quiet', action='store_true',
                            help='Flag specifying whether to hide the progress'
                                 'bar.')#
     arg_utils.add_argument('--evaluate', action='store_true',
                            help='Flag specifying whether to evaluate the Qnetwork')#
     arg_utils.add_argument('--debug', action='store_true',
                            help='Flag specifying whether the script has to be'
                                 'run in debug mode.')#
     arg_utils.add_argument('--wandb', action='store_true',
                            help='log results to wandb')#
     arg_alg.add_argument("--wandb_entity", type=str, help="Name of the entity of Wandb.")
     arg_utils.add_argument('--use_timestamp', action='store_true',
                            help='Add timestamp to the results folder.')#
     arg_utils.add_argument('--results_dir', type=str, default='logs/',
                            help='Results directory name.')#
     arg_utils.add_argument('--exp_name', type=str, default='',
                            help='Name of the experiment.')#
     arg_utils.add_argument("--n_exp", type=int)
     arg_utils.add_argument('--seed', type=int, nargs = '+', 
                            help='Seed to be used.')#

     args = parser.parse_args()

     return args