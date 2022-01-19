import os
import numpy as np
# from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_HiC_epigenetics(HiC, epis, output, fig_width=12.0,
                              vmin=0, vmax=None, cmap='Reds', colorbar=True,
                              colorbar_orientation='vertical',
                              epi_labels=None, x_ticks=None, fontsize=24,
                              epi_colors=None, epi_yaxis=True,
                              heatmap_ratio=0.6, epi_ratio=0.1,
                              interval_after_heatmap=0.05, interval_between_epi=0.01, ):
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
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        if not epi_yaxis:
            ax1.set_yticks([])
            ax1.set_yticklabels([])
        else:
            ax1.spines['right'].set_visible(True)
            ax1.tick_params(labelsize=fontsize)
            ax1.yaxis.tick_right()

        if i != len(epis) - 1:
            ax1.set_xticks([])
            ax1.set_xticklabels([])
        # ax1.axis('off')
        # ax1.xaxis.set_visible(True)
        # plt.setp(ax1.spines.values(), visible=False)
        # ax1.yaxis.set_visible(True)

        ax1.set_xlim([-0.5, N - 0.5])
        if epi_labels:
            ax1.set_ylabel(epi_labels[i], fontsize=fontsize, rotation=0)
    ax1.spines['bottom'].set_visible(True)
    if x_ticks:
        tick_pos = np.linspace(0, N - 1, len(x_ticks))  # 这个坐标其实是不对的 差1个bin 但是为了ticks好看只有先这样了
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(x_ticks, fontsize=fontsize)
    else:
        ax1.set_xticks([])
        ax1.set_xticklabels([])

    plt.savefig(output)
    plt.close()


if __name__ == '__main__':
    # Get current path
    my_path = os.path.abspath(os.path.dirname(__file__))

    #######################################################################################
    #  The below lines need to be consistent with 01_generate_example_input_regions.py
    #######################################################################################

    # Cell name
    impute_cell_name = 'HFF'

    # epigenomics names
    epi_names = ['DNase_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']

    # The regions you need to impute (must >= 250Kb)
    ch_coord = {'chr2': (23850000, 24100000)}

    # temp folder for storage
    temp_folder = f'{my_path}/temp'

    #######################################################################################
    #  The above lines are all you need to set
    #######################################################################################

    for ch in ch_coord:
        st, ed = ch_coord[ch]
        # print(ch, st, ed)
        for position in range(st, ed - 249999, 50000):
            print(ch, position)
            hic = np.load(f'{temp_folder}/example_inputs/{ch}_{position}_hic.npy')
            # print(hic.shape)
            epi = np.load(f'{temp_folder}/example_inputs/{ch}_{position}_epi.npy')
            # print(epi.shape)
            pred = np.load(f'{temp_folder}/example_outputs/{ch}_{position}_pred.npy')
            # print(pred.shape)

            visualize_HiC_epigenetics(hic, epi, f'{temp_folder}/example_outputs/{ch}_{position}_input.png',
                                      colorbar=True, interval_after_heatmap=0.,
                                      interval_between_epi=0., epi_labels=epi_names,
                                      epi_yaxis=True, fontsize=20, epi_ratio=0.045,
                                      x_ticks=['', '', '', '', '', ''], vmax=np.quantile(hic, 0.98))

            visualize_HiC_epigenetics(pred, epi, f'{temp_folder}/example_outputs/{ch}_{position}_pred.png',
                                      colorbar=True, interval_after_heatmap=0.,
                                      interval_between_epi=0., epi_labels=epi_names,
                                      epi_yaxis=True, fontsize=20, epi_ratio=0.045,
                                      x_ticks=['', '', '', '', '', ''], vmax=np.quantile(pred, 0.98))







