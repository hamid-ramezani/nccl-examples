######################Example Command#################################
'''
export WS=/mnt/c/local_storage/workspace
export WS=~/workspace
python $WS/mygitrepo/locallyasyncsgd/process_pdfs.py \
--results_group_dir . --tags TrainLoss \
TestLoss TrainAcc_1 TestAcc_1 --ncols 2 --requiredhparameters \
training_type
'''
###############################################################
import json
import pandas as pd
import os
import glob
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# sns.set_style("whitegrid", {"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=.5, rc={"grid.linewidth": 0.2})

# sns.set_style("darkgrid", {"axes.facecolor": ".9", 'axes.edgecolor': '.8','grid.color': '.6'})
# sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.width': '0.02'})

smallfontsize = 8
smallestfontsize = 7
axislabelfontsize = 8
markersize = 0.7
mpl.use('pdf')
mpl.rcParams.keys()
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 0.1
mpl.rcParams['xtick.minor.width'] = 0.1
mpl.rcParams['ytick.major.width'] = 0.1
mpl.rcParams['ytick.minor.width'] = 0.1
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=smallfontsize)
plt.rc('ytick', labelsize=smallfontsize)
plt.rc('axes', labelsize=smallfontsize)
plt.locator_params(nbins=6)
plt.rcParams.update({
    "pgf.texsystem":
    "xelatex",
    "pgf.preamble": [
        r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{amsfonts}\usepackage{amsmaths}'
    ]
})
plt.rc('font', family='sans-serif', serif='DejaVu Sans')


def get_label(name):
    if 'rbetaef_' in name:
        if 'prob25' in name:
            return r'ER$\beta.25$'
        elif 'prob50' in name:
            return r'ER$\beta.50$'
        elif 'prob75' in name:
            return r'ER$\beta.75$'
    elif 'rbeta_' in name:
        if 'prob25' in name:
            return r'R$\beta.25$'
        elif 'prob50' in name:
            return r'R$\beta.50$'
        elif 'prob75' in name:
            return r'R$\beta.75$'
    elif 'hw' in name:
        return r'HW'
    elif 'sgd' in name:
        return r'SGD'


def plot_data(datalist, trajectory_color_list, marker_size, marker_alpha,
              line_wid, marker_type_list, tagy, tagx, imagename):
    markers = [
        '<', 'H', '^', 'o', '*', '8', '^', '1', '2', '3', '4', 'X', 'd', '|'
    ]
    linestyles_e = [(0, (1, 1)), (0, (5, 5)), (0, (5, 1)), '-',
                    'dotted', 'dashdot', 'dashed', (0, (5, 2)), (0, (5, 3)),
                    (0, (5, 4)), '--']
    x_labels = [r'Epochs', r'Time (Sec)']
    y_labels = [r'Loss', r'Acc@1(\%)']
    markeveries = 10  #[200, 300]

    if 'Epoch' in tagx:
        xli = 0
    elif 'Time' in tagx:
        xli = 1

    if 'Acc' in tagy:
        yli = 1
    if 'Loss' in tagy:
        yli = 0
    print(tagy)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.12, bottom=.15, right=.99, top=.99)
    ax.set_xlabel(x_labels[xli], fontsize=axislabelfontsize)
    ax.tick_params(axis='both', direction='in', pad=1, length=1)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_ylabel(y_labels[yli], fontsize=axislabelfontsize)
    ax.yaxis.set_label_coords(-0.08, 0.42)

    for index, data in enumerate(datalist):
        on_x = [item[0] for item in data]
        on_y = [item[1] for item in data]
        ax.plot(on_x,
                on_y,
                label="(" + str(index + 1) + ")",
                color=trajectory_color_list[index %
                                            len(trajectory_color_list)],
                marker=markers[index % len(markers)],
                markevery=markeveries + 2 * index,
                linestyle=linestyles_e[index % len(linestyles_e)],
                markersize=markersize)
    (lines, labels) = plt.gca().get_legend_handles_labels()

    # if args.withlegend:
    plt.legend(lines,
               labels,
               frameon=False,
               loc='best',
               ncol=args.ncols,
               handlelength=1,
               handletextpad=0.1,
               prop={'size': smallestfontsize},
               labelspacing=0.5,
               columnspacing=0.5,
               title_fontsize=smallfontsize)
    # sns.despine(left=True, bottom=True)
    text_width = 6.75
    figs_in_a_row = 3
    fig_width = 0.4 * text_width #/ figs_in_a_row
    fig_height = 1.1 # fig_width * 0.65
    fig.set_size_inches(fig_width, fig_height)
    fig.savefig(args.results_group_dir + "/pdf_plots/" + imagename + '.pdf')


def get_data(result_json, tag, wrt):
    data = []
    for r in result_json:
        if result_json[r]['tag'] == tag:
            data.append((result_json[r][wrt], result_json[r]['val']))
    return data


def plot_tag_multi_results(args, summary_dict):
    # background_fill_color = "#fafafa"
    trajectory_colors = [
        "navy", "firebrick", "olive", "aqua", "teal", "deepskyblue",
        "darkslateblue", "blueviolet", "fuchsia", "purple", "indigo",
        "deeppink", "mediumvioletred", "darkslategray", "sienna", "maroon"
    ]
    marker_types = [
        "circle", "triangle", "square", "asterisk", "diamond",
        "inverted_triangle", "x", "cross", "circle_x", "square_x",
        "square_cross"
    ]
    for wrt, tagx in zip(['ep', 'time'], ['Epoch', 'Time (Sec)']):
        for tagy in args.tags:
            datalist = []
            imagename = wrt + tagy
            for label, rd in enumerate(args.results_dirs):
                with open(rd + "/results.json", 'r') as json_file:
                    results = json.load(json_file)
                if 'Acc_' in tagy:
                    tagy = tagy.replace('Acc_', 'Acc@')
                data = get_data(results, tagy, wrt)
                data.sort(key=itemgetter(0))
                if wrt == 'ep':
                    if "Loss" in tagy:
                        summary_dict["(" + str(label + 1) + ")"].update({
                            tagy:
                            '{:.3f}'.format(min(data, key=itemgetter(1))[1])
                        })
                    elif "Acc" in tagy:
                        summary_dict["(" + str(label + 1) + ")"].update({
                            tagy:
                            '{:.2f}'.format(max(data, key=itemgetter(1))[1])
                        })
                elif wrt == 'time':
                    summary_dict["(" + str(label + 1) + ")"].update({
                        "Time (S)":
                        '{:.0f}'.format(max(data, key=itemgetter(0))[0])
                    })
                datalist.append(data)
            plot_data(datalist, trajectory_colors, args.marker_size,
                      args.marker_alpha, args.line_wid, marker_types, tagy,
                      tagx, imagename)


def get_title_string(argdict):
    titlestring = "Arch_" + str(argdict['model'])
    titlestring = titlestring + "_Data_" + str(argdict['dataset'])
    # titlestring = titlestring + ", Schedule: " + str(argdict['scheduler_type'])
    return titlestring


def filter_argdict(argdict):
    # if 'assmswitchepochs' in argdict:
    #     argdict.update({'PASSM_St': argdict['assmswitchepochs'][0]})
    for key in [
            'averaging_type', 'averaging_sync', 'concurrency_type',
            'averaging_interval', 'num_peers'
    ]:
        if key in argdict:
            del argdict[key]

    if argdict['training_type'] in ['Seq', 'ddp']:
        for key in [
                'num_processes', 'averaging_type', 'averaging_sync',
                'concurrency_type', 'averaging_interval', 'num_peers'
        ]:
            if key in argdict:
                del argdict[key]
    elif argdict['training_type'] in [
            'Mp', 'PASSM', 'ASSM', 'P_PASSM', 'lfP_PASSM', 'ddp', 'dP_PASSM'
    ]:
        for key in [
                'averaging_type', 'averaging_sync', 'concurrency_type',
                'averaging_interval', 'num_peers'
        ]:
            if key in argdict:
                del argdict[key]
    if argdict['training_type'] not in ['RBeta', 'EFRBeta']:
        for key in ['beta']:
            if key in argdict:
                del argdict[key]

    if argdict['training_type'] not in ['P_PASSM', 'lfP_PASSM', 'dP_PASSM']:
        for key in ['assmswitchepochs', 'dampen_for_passm']:
            if key in argdict:
                del argdict[key]


def filter_table(tex_table):
    tex_table = tex_table.replace("cifar100", "C100")
    tex_table = tex_table.replace("cifar10", "C10")
    tex_table = tex_table.replace("num\\_processes", "$\mathbf{U}$")
    tex_table = tex_table.replace("averaging\\_freq", "$\mathbf{K}$")
    tex_table = tex_table.replace("training\\_type", "\\textbf{Method}")
    tex_table = tex_table.replace("train\\_bs", "$\mathbf{B}$")
    tex_table = tex_table.replace("commsize", "$\mathbf{Q}$")
    tex_table = tex_table.replace("Hw", "\hw")
    tex_table = tex_table.replace("mnalsgPHW", "\\palsgd")
    tex_table = tex_table.replace("mnalsgHW", "\\alsgd")
    tex_table = tex_table.replace("Seq", "\\sgd")
    tex_table = tex_table.replace("mnddp", "\\mbsgd")
    tex_table = tex_table.replace("mnlSGD", "\\plsgd")
    tex_table = tex_table.replace("ddp", "\\sgd")
    tex_table = tex_table.replace("TrainLoss", "\\textbf{Tr.L.}")
    tex_table = tex_table.replace("TestLoss", "\\textbf{Te.L.}")
    tex_table = tex_table.replace("TrainAcc@1", "\\textbf{Tr.A.}")
    tex_table = tex_table.replace("TestAcc@1", "\\textbf{Te.A.}")
    tex_table = tex_table.replace("Time (S)", "$T$")
    # tex_table = tex_table.replace("TrainLoss",
    #                               "\\textbf{Train{\\newline}Loss}")
    # tex_table = tex_table.replace("TestLoss", "\\textbf{Test{\\newline}Loss}")
    # tex_table = tex_table.replace("TrainAcc@1",
    #                               "\\textbf{Train{\\newline}Acc.}")
    # tex_table = tex_table.replace("TestAcc@1", "\\textbf{Test{\\newline}Acc.}")
    # tex_table = tex_table.replace("Time (S)", "\\textbf{Time{\\newline}(S)}")
    return tex_table


def process_multi_results_plots(args, plot_dir=None):
    if type(args.results_dirs) is not list:
        print("The results directories should be a list!")
    summary_dict = {}
    argdicts = []
    for label, results_dir in enumerate(args.results_dirs):
        with open(results_dir + "/args.json", 'r') as json_file:
            argdict = json.load(json_file)
        argdicts.append(argdict)
        # print(argdict.items())
        # legend_string = get_legend_string(argdict)
        hp_dict = dict(
            filter(lambda elem: elem[0] in args.requiredhparameters,
                   argdict.items()))
        filter_argdict(hp_dict)
        summary_dict.update({"(" + str(label + 1) + ")": {}})
        summary_dict["(" + str(label + 1) + ")"].update(hp_dict)
    plot_tag_multi_results(args, summary_dict)
    # indices = [
    #     'model', 'training_type', 'dataset', 'train_bs', 'averaging_freq',
    #     'num_processes', 'TrainLoss', 'TestLoss', 'TrainAcc@1', 'TestAcc@1',
    #     'Time (S)'
    # ]
    for ind in args.indices:
        if ind not in args.tags and ind not in args.requiredhparameters and ind not in [
                'TrainAcc@1',
                'TestAcc@1',
                'Time (S)',
        ]:
            args.indices.remove(ind)
    df = pd.DataFrame(index=args.indices, data=summary_dict)
    df = df.fillna('--').T
    tex_table = df.to_latex()
    tex_table = filter_table(tex_table)
    table_writer = open(
        args.results_group_dir + "/pdf_plots/" + get_title_string(argdict) +
        '.tex', "w")
    table_writer.write(tex_table)
    table_writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bapi Training')
    parser.add_argument(
        '--tags',
        nargs='+',
        type=str,
        default=[  #, "TrainAcc_5", "TestAcc_5", "LR"
            "TrainLoss", "TrainAcc_1", "TestLoss", "TestAcc_1"
        ],
        help='Tags')
    parser.add_argument(
        '--requiredhparameters',
        nargs='+',
        type=str,
        default=[  #, "TrainAcc_5", "TestAcc_5", "LR"
            'training_type',
            'lr',
            'train_bs',  #'bs_multiple',
            'epochs',
            'num_processes',
            'averaging_type',
            'averaging_sync',
            'concurrency_type',
            'averaging_interval',
            'assmswitchepochs',
            'num_peers',
            'dampen_for_passm',
            'beta',
            'dataset'
        ],
        help='Tags')
    parser.add_argument(
        '--indices',
        nargs='+',
        type=str,
        default=[  # 'model', 'dataset',   'averaging_freq',   
            'training_type', 'commsize', 'train_bs', 'num_processes',
            'TrainLoss', 'TrainAcc@1', 'TestLoss', 'TestAcc@1', 'Time (S)'
        ],
        help='Tags')
    parser.add_argument('--results_dirs',
                        nargs='+',
                        type=str,
                        default=None,
                        help='Directories')
    parser.add_argument('--results_group_dir',
                        default='.',
                        type=str,
                        help='Parent Directories')
    parser.add_argument('--plot_height', type=int, default=250)
    parser.add_argument('--plot_width', type=int, default=250)
    parser.add_argument('--ncols', type=int, default=2)
    parser.add_argument('--num_markers', type=int, default=10)
    parser.add_argument('--line_wid', type=int, default=1)
    parser.add_argument('--marker_alpha', type=float, default=0.5)
    parser.add_argument('--marker_size', type=int, default=8)
    parser.add_argument('--legend_fig_height', type=int, default=100)
    parser.add_argument('--legend_fig_width', type=int, default=80)
    args = parser.parse_args()
    if args.results_dirs is None:
        args.results_dirs = glob.glob(args.results_group_dir + '/results*')
    if not os.path.exists(args.results_group_dir + "/pdf_plots/"):
        os.makedirs(args.results_group_dir + "/pdf_plots/")
    if len(args.results_dirs) >= 1:
        process_multi_results_plots(args, plot_dir=args.results_group_dir)
    else:
        print("Not enough result dirs!")

