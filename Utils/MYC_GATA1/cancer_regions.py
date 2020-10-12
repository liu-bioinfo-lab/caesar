import numpy as np
import os
from scipy.stats import zscore
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model import model_fn, model_loop
import argparse


MOUSE_CHR_SIZES = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                   'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                   'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                   'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                   'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                   'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                   'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                   }

HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }


def load_chrom_sizes(reference_genome):
    """
    Load chromosome sizes for a reference genome
    """
    my_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(my_path, reference_genome + '.chrom.sizes'))
    lengths = {}
    for line in f:
        [ch, l] = line.strip().split()
        lengths[ch] = int(l)
    return lengths


def load_epigenetic_data(chromosomes, epi_names, epi_path, cell_type='GM12878', verbose=1):
    """
    Load epigenetic data from processed file (.npy files)

    Args:
        epi_names (list): the folder for storing data from different chromosomes
        verbose (int):

    Return:
         epigenetic_data (dict): {chromosome: numpy.array (shape: n_bins * n_channels)}
    """

    epigenetic_data = {}
    res = 200

    for ch in chromosomes:
        epigenetic_data[ch] = None
        for i, k in enumerate(epi_names):
            path = f'{epi_path}/{cell_type}_{ch}_{res}bp_{k}.npy'
            s = np.load(path)
            # print(ch, k, s.shape)
            s = zscore(s)
            if verbose:
                print(cell_type, ch, k, len(s))
            if i == 0:
                epigenetic_data[ch] = np.zeros((len(s), len(epi_names)))
            epigenetic_data[ch][:, i] = s
            # epigenetic_data[ch] = epigenetic_data[ch].T
    return epigenetic_data


def load_epigenetic_data2(chromosomes, epi_names, epi_path, cell_type='GM12878', verbose=1):
    """
    Load epigenetic data from processed file (.npy files)

    Args:
        epi_names (list): the folder for storing data from different chromosomes
        verbose (int):

    Return:
         epigenetic_data (dict): {chromosome: numpy.array (shape: n_bins * n_channels)}
    """

    epigenetic_data = {}
    res = 200

    for ch in chromosomes:
        epigenetic_data[ch] = None
        for i, k in enumerate(epi_names):
            path = f'{epi_path}/{cell_type}_{ch}_{res}bp_{k}.npy'
            s = np.load(path)
            # print(ch, k, s.shape)
            # s = zscore(s)
            s = s / np.mean(s)
            if verbose:
                print(cell_type, ch, k, len(s))
            if i == 0:
                epigenetic_data[ch] = np.zeros((len(s), len(epi_names)))
            epigenetic_data[ch][:, i] = s
            # epigenetic_data[ch] = epigenetic_data[ch].T
    return epigenetic_data


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cell_line',
        type=str,
        default='hESC',
        help='cell_type'
    )
    parser.add_argument(
        '--chrs',
        type=str,
        default='1,4,7,10,13,17,18',
        help='"All", "all" or comma-separated chromosome ids'
    )
    parser.add_argument(
        '--inp_resolutions',
        type=str,
        default='1000',
        help='comma-separated input resolutions'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=8,
        help='number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='batch size'
    )
    parser.add_argument(
        '--n_GC_layers',
        type=int,
        default=2,
        help='number of GC layers'
    )
    parser.add_argument(
        '--n_GC_units',
        type=int,
        default=96,
        help='number of hidden units in GC layers'
    )
    parser.add_argument(
        '--inp_window',
        type=int,
        default=15,
        help='window size of the input conv layer'
    )
    parser.add_argument(
        '--inp_kernel',
        type=int,
        default=96,
        help='number of kernel of the input first conv layer'
    )
    parser.add_argument(
        '--conv_kernels',
        type=str,
        default='96',
        help='comma-separated numbers of conv kernels'
    )
    parser.add_argument(
        '--conv_windows',
        type=str,
        default='15',
        help='comma-separated numbers of conv windows'
    )
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=1,
        help='how frequent to save model'
    )
    parser.add_argument(
        '--n_marks',
        type=int,
        default=6,
        help='number of epigenetic marks'
    )
    args, _ = parser.parse_known_args()
    return args


def load_model(args, weights):
    model = model_fn(
        first_layer=[args.inp_kernel, args.inp_window],
        gcn_layers=[args.n_GC_units] * args.n_GC_layers,
        conv_layer_filters=[int(k) for k in args.conv_kernels.split(',')],
        conv_layer_windows=[int(k) for k in args.conv_windows.split(',')],
        nBins=1250,
        lr=args.lr,
        nMarks=args.n_marks,
        verbose=1
    )

    model.load_weights(weights)
    return model


def visualize_HiC_epigenetics(HiC, epis, output, fig_width=12.0,
                              vmin=0, vmax=None, cmap='Reds', colorbar=True,
                              colorbar_orientation='vertical',
                              epi_labels=None, x_ticks=None, fontsize=24,
                              epi_colors=None, epi_yaxis=True,
                              heatmap_ratio=0.6, epi_ratio=0.1,
                              interval_after_heatmap=0.05, interval_between_epi=0.01,):
    """
    Visualize matched HiC and epigenetic signals in one figure
    Args:
        HiC (numpy.array): Hi-C contact map, only upper triangle is used.
        epis (list): epigenetic signals
        output (str): the output path. Must in a proper format (e.g., 'png', 'pdf', 'svg', ...).
        fig_width (float): the width of the figure. Then the height will be automatically calculated. Default: 12.0
        vmin (float): min value of the colormap. Default: 0
        vmax (float): max value of the colormap. Will use the max value in Hi-C data if not specified.
        cmap (str or plt.cm): which colormap to use. Default: 'Reds'
        colorbar (bool): whether to add colorbar for the heatmap. Default: True
        colorbar_orientation (str): "horizontal" or "vertical". Default: "vertical"
        epi_labels (list): the names of epigenetic marks. If None, there will be no labels at y axis.
        x_ticks (list): a list of strings. Will be added at the bottom. THE FIRST TICK WILL BE AT THE START OF THE SIGNAL, THE LAST TICK WILL BE AT THE END.
        fontsize (int): font size. Default: 24
        epi_colors (list): colors of epigenetic signals
        epi_yaxis (bool): whether add y-axis to epigenetic signals. Default: True
        heatmap_ratio (float): the ratio of (heatmap height) and (figure width). Default: 0.6
        epi_ratio (float): the ratio of (1D epi signal height) and (figure width). Default: 0.1
        interval_after_heatmap (float): the ratio of (interval between heatmap and 1D signals) and (figure width). Default: 0.05
        interval_between_epi (float): the ratio of (interval between 1D signals) and (figure width). Default: 0.01

    No return. Save a figure only.
    """

    # Make sure the lengths match
    # len_epis = [len(epi) for epi in epis]
    # if max(len_epis) != min(len_epis) or max(len_epis) != len(HiC):
    #     raise ValueError('Size not matched!')
    N = len(HiC)

    # Define the space for each row (heatmap - interval - signal - interval - signal ...)
    rs = [heatmap_ratio, interval_after_heatmap] + [epi_ratio, interval_between_epi] * len(epis)
    rs = np.array(rs[:-1])

    # Calculate figure height
    fig_height = fig_width * np.sum(rs)
    rs = rs / np.sum(rs)  # normalize to 1 (ratios)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Split the figure into rows with different heights
    gs = GridSpec(len(rs), 1, height_ratios=rs)

    # Ready for plotting heatmap
    ax0 = plt.subplot(gs[0, :])
    # Define the rotated axes and coordinates
    coordinate = np.array([[[(x + y) / 2, y - x] for y in range(N + 1)] for x in range(N + 1)])
    X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
    # Plot the heatmap
    vmax = vmax if vmax is not None else np.max(HiC)
    im = ax0.pcolormesh(X, Y, HiC, vmin=vmin, vmax=vmax, cmap=cmap)
    ax0.axis('off')
    ax0.set_ylim([0, N])
    ax0.set_xlim([0, N])
    if colorbar:
        if colorbar_orientation == 'horizontal':
            _left, _width, _bottom, _height = 0.12, 0.25, 1 - rs[0] * 0.25, rs[0] * 0.03
        elif colorbar_orientation == 'vertical':
            _left, _width, _bottom, _height = 0.9, 0.02, 1 - rs[0] * 0.7, rs[0] * 0.5
        else:
            raise ValueError('Wrong orientation!')
        cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                            orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.outline.set_visible(False)

    # print(rs/np.sum(rs))
    # Ready for plotting 1D signals
    if epi_labels:
        assert len(epis) == len(epi_labels)
    if epi_colors:
        assert len(epis) == len(epi_colors)

    for i, epi in enumerate(epis):
        # print(epi.shape)
        ax1 = plt.subplot(gs[2 + 2 * i, :])
        if epi_colors:
            ax1.fill_between(np.arange(N), 0, epi, color=epi_colors[i])
        else:
            ax1.fill_between(np.arange(N), 0, epi)
        if not epi_yaxis:
            ax1.set_yticks([])
            ax1.set_yticklabels([])
            ax1.spines['left'].set_visible(False)
        else:
            ax1.tick_params(labelsize=fontsize)
        if i != len(epis) - 1:
            ax1.set_xticks([])
            ax1.set_xticklabels([])
        # ax1.axis('off')
        # ax1.xaxis.set_visible(True)
        # plt.setp(ax1.spines.values(), visible=False)
        # ax1.yaxis.set_visible(True)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xlim([-0.5, N-0.5])
        if epi_labels:
            ax1.set_ylabel(epi_labels[i], fontsize=fontsize, rotation=0)
    ax1.spines['bottom'].set_visible(True)
    if x_ticks:
        tick_pos = np.linspace(0, N - 1, len(x_ticks))   # 这个坐标其实是不对的 差1个bin 但是为了ticks好看只有先这样了
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(x_ticks, fontsize=fontsize)
    else:
        ax1.set_xticks([])
        ax1.set_xticklabels([])

    plt.savefig(output)
    plt.close()


def work(cell_lines, region='MYC'):
    # MYC:
    if region == 'MYC':
        ch = 'chr8'
        pos = 127600000
        st, ed = 0, 1250

    # GATA1
    elif region == 'GATA1':
        ch = 'chrX'
        pos = 48725000
        st, ed = 0, 500

    else:
        raise ValueError

    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']

    model1 = load_model(args(), 'HFF6_temp_model_49.h5')
    model2 = model_loop()
    model2.load_weights('HFF6_loop_model_49.h5')

    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'

    for cell_line in cell_lines:
        epi_data = load_epigenetic_data(
            [ch], epi_names,
            '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/cancer',
            cell_type=cell_line, verbose=1
        )
        epi_orig = load_epigenetic_data2(
            [ch], epi_names,
            '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/cancer',
            cell_type=cell_line, verbose=1
        )

        print(cell_line, pos)
        hic = np.load(f'{path}/HiC/mix/{ch}/{ch}_1000bp_{pos}_{pos + 250000}.npy')
        epi = epi_data[ch][pos // 200 - 7: pos // 200 + 1257, :]
        epi2 = epi_orig[ch][pos // 200: pos // 200 + 1250, :].T
        hics = np.array([hic])
        epis = np.array([epi])

        pred1 = model1.predict([hics, epis])[0, :, :]
        pred2 = model2.predict([hics, np.array([epi_data[ch][pos // 200 - 1: pos // 200 + 1251, :]])])[0, :, :]
        for x1 in range(1250):
            for x2 in range(-20, 21):
                if 0 <= x1 + x2 < 1250:
                    pred2[x1, x1 + x2] = 0
        pred2[pred2 < np.quantile(pred2, 0.99)] = 0
        pred2 = convolve2d(pred2, np.ones((3, 3)) / 9, mode='same')
        pred = pred1 + pred2 * 0.2

        # names = ['ATAC-seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
        pred = convolve2d(pred, np.ones((3, 3)) / 9, mode='same')
        pred = np.exp(pred) - 1
        pred = pred[st:ed, st:ed]
        epi_or = epi2[:, st:ed]
        np.save(f'mat_{cell_line}_{region}_pred.npy', pred)

        visualize_HiC_epigenetics(pred, epi_or, f'{cell_line}_{region}_pred.png', colorbar=True,
                                  interval_after_heatmap=0.,
                                  interval_between_epi=0., epi_labels=epi_names,
                                  epi_yaxis=False, fontsize=20, epi_ratio=0.045,
                                  x_ticks=[str(pos), '', '', '', '', str(pos + ed * 200)], vmax=np.quantile(pred, 0.99)
                                  )

        np.save(f'epi_{cell_line}_{region}.npy', epi_or)


# work(['KMS-11', 'MM.1S', 'Karpas-422', 'GM12878', 'OCI-LY7', 'K562'], 'MYC')
# work(['KMS-11', 'MM.1S', 'Karpas-422', 'GM12878', 'OCI-LY7', 'K562'], 'GATA1')
# work(['Karpas-422', 'K562'], 'MYC')
# work(['Karpas-422', 'K562'], 'GATA1')
work(['K562'], 'MYC')
work(['K562'], 'GATA1')
