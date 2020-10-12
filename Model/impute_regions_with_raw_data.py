import numpy as np
import os
import pyBigWig
from scipy.stats import zscore
from scipy.signal import convolve2d
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model_util import model_fn, model_loop


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


def load_bigwig_for_one_region(path, chromosome, start_pos, end_pos, resolution, output_path=None):
    nBins = (end_pos - start_pos) // resolution
    bw = pyBigWig.open(path)
    vec = bw.stats(chromosome, start_pos, end_pos, exact=True, nBins=nBins)
    for i in range(len(vec)):
        if vec[i] is None:
            vec[i] = 0
    if output_path is not None:
        np.save(output_path, vec)
    return vec


def load_bedGraph_for_one_region(path, chromosome, start_pos, end_pos, resolution, output_path=None, score_column=4):
    # score_column: which column is score (The column indices start from 1.)
    assert start_pos % resolution == 0
    assert end_pos % resolution == 0
    nBins = (end_pos - start_pos) // resolution
    epi_signal = np.zeros((nBins, ))

    f = open(path)
    for line in f:
        lst = line.strip().split()
        ch, p1, p2, v = lst[0], int(lst[1]), int(lst[2]), float(lst[score_column - 1])
        if ch != chromosome or p1 < start_pos or p2 >= end_pos:
            continue
        pp1, pp2 = p1 // resolution, int(np.ceil(p2 / resolution))
        for i in range(pp1, pp2):
            value = (min(p2, (i + 1) * resolution) - max(p1, i * resolution)) / resolution * v
            epi_signal[i - start_pos // resolution] += value
    if output_path is not None:
        np.save(output_path, epi_signal)
    return epi_signal


def normalize_HiC(hic, observed_exps):
    exps = np.loadtxt('HiC_exps.txt')
    for i in range(len(hic)):
        for j in range(len(hic)):
            hic[i, j] = hic[i, j] / observed_exps[abs(i - j)] * exps[abs(i - j)]
    return hic


def load_model_1(nMarks, weights):
    model = model_fn(
        first_layer=[96, 15],
        gcn_layers=[96] * 2,
        conv_layer_filters=[96],
        conv_layer_windows=[15],
        nBins=1250,
        lr=0.004,
        nMarks=nMarks,
        verbose=1
    )

    model.load_weights(weights)
    return model


def load_model_2(nMarks, weights):
    model = model_loop(
        first_layer=[96, 3],
        gcn_layers=[96] * 2,
        conv_layer_filters=[96],
        conv_layer_windows=[3],
        nBins=1250,
        lr=0.004,
        nMarks=nMarks,
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
    plt.show()
    plt.close()


def predict_region(
        ch, pos, hic_resolution=1000,
        hic_file=None, bigwig_files=None, bedgraph_files=None
):
    assert hic_file is not None
    assert (bigwig_files is not None) or (bedgraph_files is not None)
    # Settings
    max_distance = 250000
    resolution = 200
    length = load_chrom_sizes('hg38')[ch]
    window1, window2 = 15, 3  # the convolution window size for the two models
    padding1, padding2 = window1 // 2, window2 // 2
    dim = max_distance // resolution

    # Load epigenomic features
    print('Loading epigenomic features...')
    epi_names = ['DNase-seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    epi = np.zeros((dim + 2 * padding1, len(epi_names)))
    epi_original = np.zeros((len(epi_names), dim))
    st, ed = pos // resolution, (pos + max_distance) // resolution

    for i, epi_name in enumerate(epi_names):
        print(f' - {epi_name}')
        if bigwig_files is not None:
            signal = load_bigwig_for_one_region(
                bigwig_files[i], ch,
                0, length,
                resolution
            )
            epi_original[i, :] = signal[st:ed]
            epi[:, i] = zscore(signal)[st-padding1:ed+padding1]
        else:
            # Assume the 4-th column is the score
            signal = load_bedGraph_for_one_region(
                bigwig_files[i], ch,
                0, length,
                resolution, score_column=4)
            epi_original[i, :] = signal[st:ed]
            epi[:, i] = zscore(signal)[st - padding1:ed + padding1]

    # Load Hi-C contact map
    print('Loading the Hi-C contact map...')
    fold = hic_resolution // resolution
    hic = np.zeros((dim // fold, dim // fold))
    count = 0
    strata_sum = np.zeros((dim // fold,))  # sum of each strata
    for line in open(hic_file):
        if count % 5000000 == 0:
            print(f' - Line: {count}')
        count += 1
        lst = line.strip().split()
        p1, p2, v = int(lst[0]), int(lst[1]), float(lst[2])
        pp1, pp2 = (p1 - pos) // hic_resolution, (p2 - pos) // hic_resolution
        if abs(pp1 - pp2) < dim // fold:
            strata_sum[abs(pp1 - pp2)] += v
        if max(pp1, pp2) < dim // fold and min(pp1, pp2) >= 0:
            hic[pp1, pp2] += v
            if pp1 != pp2:
                hic[pp2, pp1] += v
    strata_mean = [elm / (length // hic_resolution + 1 - i) for i, elm in enumerate(strata_sum)]
    # print(strata_mean[:30])
    hic = normalize_HiC(hic, strata_mean)
    fc_ = 1 / fold
    f = interp2d(np.arange(dim // fold), np.arange(dim // fold), hic)
    new_co = np.linspace(-0.5 + fc_ / 2, dim // fold - 0.5 - fc_ / 2, dim)
    hic = f(new_co, new_co)
    hic = np.log(hic + 1)

    # Load model
    print('Loading models...')
    model1 = load_model_1(nMarks=len(epi_names), weights='contact_profile_model_49.h5')
    model2 = load_model_2(nMarks=len(epi_names), weights='loop_model_45.h5')

    # Predict
    print('Predicting...')
    msk = np.ones((dim, dim))  # mask near-diagonal entries for loop model's output
    for i in range(dim):
        for j in range(max(0, i - 20), min(dim, i + 21)):
            msk[i, j] = 0

    pred1 = model1.predict(
        [np.array([hic]), np.array([epi])]
    )[0, :, :]
    pred2 = model2.predict(
        [np.array([hic]), np.array([epi[padding1-padding2:padding2-padding1]]), np.array([msk])]
    )[0, :, :] / 2
    pred2[pred2 < np.quantile(pred2, 0.995)] = 0  # remove loop predictor's noise
    pred = convolve2d(pred1 + pred2, np.ones((5, 5)) / 25, mode='same')
    pred = np.exp(pred) - 1
    visualize_HiC_epigenetics(pred, epi_original, f'outputs/{ch}_{pos}_pred.png', colorbar=True,
                              interval_after_heatmap=0.,
                              interval_between_epi=0., epi_labels=epi_names,
                              epi_yaxis=False, fontsize=20, epi_ratio=0.045,
                              x_ticks=[str(pos), '', '', '', '', str(pos + 250000)], vmax=np.quantile(pred, 0.99)
                              )

    visualize_HiC_epigenetics(hic, epi_original, f'outputs/{ch}_{pos}_hic.png', colorbar=True,
                              interval_after_heatmap=0.,
                              interval_between_epi=0., epi_labels=epi_names,
                              epi_yaxis=False, fontsize=20, epi_ratio=0.045,
                              x_ticks=[str(pos), '', '', '', '', str(pos + 250000)], vmax=np.quantile(hic, 0.99)
                              )


if __name__ == '__main__':
    hic_path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/Dekker_HFF/processed/chr2_1kb.txt'
    epi_names = ['DNase_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    epi_path = [
        f'/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/human_tissues/K562/K562_{i}_hg38.bigWig' for i in epi_names
    ]

    predict_region(
        ch='chr2', pos=70100000, hic_resolution=1000,
        hic_file=hic_path,
        bigwig_files=epi_path
    )

