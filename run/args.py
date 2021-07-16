import argparse


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm',
                        type=str,
                        nargs='*')
    parser.add_argument('--environment', '-e',
                        type=str,
                        nargs='*',
                        default=[''])
    parser.add_argument('--directory', '-d',
                        type=str,
                        default='',
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--kwargs', '-kw',
                        type=str,
                        nargs='*',
                        default=[])
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='directory prefix')
    parser.add_argument('--model-name', '-n',
                        default='',
                        help='model name')
    parser.add_argument('--logdir', '-ld',
                        type=str,
                        default='logs')
    parser.add_argument('--grid-search', '-gs',
                        action='store_true')
    parser.add_argument('--delay',
                        default=1,
                        type=int)
    parser.add_argument('--verbose', '-v',
                        type=str,
                        default='warning')
    args = parser.parse_args()

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        help='directory where checkpoints and "config.yaml" exist')
    parser.add_argument('--record', '-r', action='store_true')
    parser.add_argument('--video_len', '-v', type=int, default=None)
    parser.add_argument('--n_episodes', '-n', type=int, default=1)
    parser.add_argument('--n_envs', '-ne', type=int, default=0)
    parser.add_argument('--n_workers', '-nw', type=int, default=0)
    parser.add_argument('--size', '-s', nargs='+', type=int, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--force_envvec', '-fe', action='store_true')
    args = parser.parse_args()

    return args
