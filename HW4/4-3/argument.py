def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--actor_learning_rate', type=float, default=0.00025)
    parser.add_argument('--critic_learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--save_dir', type=str, default='save_a2c_EPISLON_ppo/')
    parser.add_argument('--load_saver', type=int, default=0) 
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--dueling_dqn', type=int, default=0)
    parser.add_argument('--replace_num', type=int, default=2500)
    parser.add_argument('--update_num', type=int, default=4)
    parser.add_argument('--epsilon_decay_constant', type=float, default=0.0000009)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--saver_steps', type=int, default=25000)
    parser.add_argument('--output_logs', type=str, default='ppo_EPISLON_loss.csv')
    parser.add_argument('--training_steps', type=int, default=10)
    parser.add_argument('--if_epsilon',type=int, default=0)
    return parser
