def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--epoch', type=int, default=8000)
    parser.add_argument('--N', type=int, default=21)
    parser.add_argument('--load',type=int, default=0)
    parser.add_argument('--lr',type=float, default=0.0005)
    parser.add_argument('--name',type=str, default="pong_1000_0.0005")
    parser.add_argument('--save_path',type=str, default="../models/")
    parser.add_argument('--gl_a',type=float, default=1000.0)
    parser.add_argument('--argmax',type=int, default=0)
    return parser
