import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_data(data, x, y, outdir, tag, title, timing=None):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
        if timing:
            data = data[data.Timing == timing].drop('Timing', axis=1)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.set_palette('Set2') # or husl
    if 'Timing' in data.columns:
        sns.lineplot(x=x, y=y, ax=ax, data=data, hue=tag, style='Timing')
    else:
        sns.lineplot(x=x, y=y, ax=ax, data=data, hue=tag)
    ax.grid(True, alpha=0.8, linestyle=':')
    ax.legend(loc='best').set_draggable(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if timing:
        title = f'{title}-{timing}'
    outpath = f'{outdir}/{title}.png'
    ax.set_title(title)
    fig.savefig(outpath)
    print(f'Plot Path: {outpath}')

def get_datasets(filedir, tag, condition=None):
    unit = 0
    datasets = []
    for root, _, files in os.walk(filedir):
        for f in files:
            if f.endswith('log.txt'):
                log_path = os.path.join(root, f)
                data = pd.read_csv(log_path, sep='\t')

                data.insert(len(data.columns), tag, condition)

                datasets.append(data)
                unit +=1

    return datasets

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--title', '-t', default='', type=str)
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--legendtag', '-tag', default='Algo')
    parser.add_argument('--x', '-x', default='env_step', nargs='*')
    parser.add_argument('--y', '-y', default='score', nargs='*')
    parser.add_argument('--timing', default=None, choices=['Train', 'Eval', None], 
                        help='select timing to plot; both training and evaluation stats are plotted by default')
    args = parser.parse_args()

    # by default assume using `python utility/plot.py` to call this file
    if len(args.logdir) != 1:
        dirs = [f'{d}' for d in args.logdir]
    else:
        dirs = glob.glob(args.logdir[0])

    # dir follows pattern: logs/env/algo(/model_name)
    title = args.title or dirs[0].split('/')[1].split('_')[-1]
    # set up legends
    if args.legend:
        assert len(args.legend) == len(dirs), (
            "Must give a legend title for each set of experiments: "
            f"#legends({args.legend}) != #dirs({args.dirs})")
        legends = args.legend
    else:
        legends = [path.split('/')[2] for path in dirs]
        legends = [l[3:] if l.startswith('GS-') else l for l in legends]
    tag = args.legendtag

    print('Directories:')
    for d in dirs:
        print(f'\t{d}')
    print('Legends:')
    for l in legends:
        print(f'\t{l}')
    data = []
    for logdir, legend_title in zip(dirs, legends):
        data += get_datasets(logdir, tag, legend_title)

    xs = args.x if isinstance(args.x, list) else [args.x]
    ys = args.y if isinstance(args.y, list) else [args.y]
    for x in xs:
        for y in ys:
            outdir = f'results/{title}-{x}-{y}'
            plot_data(data, x, y, outdir, tag, title, args.timing)

if __name__ == '__main__':
    main()
