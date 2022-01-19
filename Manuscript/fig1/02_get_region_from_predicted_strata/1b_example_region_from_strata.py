import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }


def visualize_HiC_triangle(HiC, output, fig_size=(12, 6.5),
                           vmin=0, vmax=None, cmap='Reds', colorbar=True,
                           colorbar_orientation='vertical',
                           x_ticks=None, fontsize=24):
    """
        Visualize matched HiC and epigenetic signals in one figure
        Args:
            HiC (numpy.array): Hi-C contact map, only upper triangle is used.
            output (str): the output path. Must in a proper format (e.g., 'png', 'pdf', 'svg', ...).
            fig_size (tuple): (width, height). Default: (12, 8)
            vmin (float): min value of the colormap. Default: 0
            vmax (float): max value of the colormap. Will use the max value in Hi-C data if not specified.
            cmap (str or plt.cm): which colormap to use. Default: 'Reds'
            colorbar (bool): whether to add colorbar for the heatmap. Default: True
            colorbar_orientation (str): "horizontal" or "vertical". Default: "vertical"
            x_ticks (list): a list of strings. Will be added at the bottom. THE FIRST TICK WILL BE AT THE START OF THE SIGNAL, THE LAST TICK WILL BE AT THE END.
            fontsize (int): font size. Default: 24

        No return. Save a figure only.
        """
    N = len(HiC)
    coordinate = np.array([[[(x + y) / 2, y - x] for y in range(N + 1)] for x in range(N + 1)])
    X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
    vmax = vmax if vmax is not None else np.max(HiC)

    fig, ax = plt.subplots(figsize=fig_size)
    im = plt.pcolormesh(X, Y, HiC, vmin=vmin, vmax=vmax, cmap=cmap)
    # plt.axis('off')
    plt.yticks([], [])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if x_ticks:
        tick_pos = np.linspace(0, N, len(x_ticks))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(x_ticks, fontsize=fontsize)
    else:
        ax.spines['bottom'].set_visible(False)

    plt.ylim([0, N])
    plt.xlim([0, N])

    if colorbar:
        if colorbar_orientation == 'horizontal':
            _left, _width, _bottom, _height = 0.7, 0.25, 0.75, 0.03
        elif colorbar_orientation == 'vertical':
            _left, _width, _bottom, _height = 0.9, 0.02, 0.3, 0.5
        else:
            raise ValueError('Wrong orientation!')
        cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                            orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.outline.set_visible(False)

    plt.savefig(output)
    # plt.show()


def strata2mat(stripe_path, loop_path, ch, start_pos, end_pos):
    id_size = (end_pos - start_pos) // 200
    curr_id = (start_pos - HUMAN_CHR_SIZES[ch][0]) // 200
    mat = np.zeros((id_size, id_size))
    for j in range(id_size):
        print(' Loading:', j)
        # method 2
        s1 = np.load(f'{stripe_path}/{ch}/strata_{j}.npy')
        s2 = np.load(f'{loop_path}/{ch}/strata_{j}.npy') * 0.7
        strata = s1 + s2
        slc = strata[curr_id: curr_id + id_size - j]
        for k in range(len(slc)):
            mat[k, k + j] = slc[k]
            if j != 0:
                mat[k + j, k] = slc[k]
    return mat


if __name__ == '__main__':
    # 1b
    ch, start_pos, end_pos = 'chr2', 70100000, 70350000
    mat = strata2mat(
        stripe_path='/nfs/turbo/umms-drjieliu/usr/temp_Fan/05_HFF_stripe_denoise_all',
        loop_path='/nfs/turbo/umms-drjieliu/usr/temp_Fan/07_HFF_loop_all',
        ch=ch, start_pos=start_pos, end_pos=end_pos
    )
    mat = convolve2d(mat, np.ones((3, 3)) / 9, mode='same')
    np.save('1b_mat.npy', mat)
    visualize_HiC_triangle(mat, '1b_v3.png', fig_size=(12, 6.5),
                           vmin=0, vmax=np.quantile(mat, 0.98), cmap='Reds', colorbar=True,
                           colorbar_orientation='vertical',
                           x_ticks=['', '', '', '', '', ''], fontsize=24)




