import numpy as np
import os
from scipy.stats import zscore
from copy import deepcopy
from keras.layers import Input, Concatenate, Permute, Conv1D, Add, BatchNormalization, Dot
from keras.models import Model
from keras.optimizers import Adam
from model import model_fn
import argparse
from scipy.interpolate import interp2d
from scipy.signal import convolve2d
from hicrep2 import hicrep


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


def load_all_strata(ch, res=1000, n_strata=250, reference='hg38'):
    lengths = load_chrom_sizes(reference)

    reads = {ch: [0 for _ in range(4)] for ch in lengths}
    for line in open('total_reads.txt'):
        idx = {'hESC': 0, 'K562': 1, 'HFF': 2, 'GM12878': 3}
        lst = line.strip().split()
        if lst[1] in idx:
            reads[lst[0]][idx[lst[1]]] = float(lst[2])
    n_reads = reads[ch]
    print(n_reads)
    n_reads = [elm / np.mean(n_reads) * len(n_reads) for elm in n_reads]

    strata = [np.zeros((lengths[ch] // res + 1 - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata for {0} at {1} resolution ...'.format(ch, res))

    names = ['hESC', 'K562', 'HFF', 'GM12878']
    chrom_files = [
        f'/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/H1-hESC/processed/{ch}_1kb.txt',
        f'/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/K562/processed/{ch}_1kb.txt',
        f'/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/Dekker_HFF/processed/{ch}_1kb.txt',
        f'/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/GM12878/processed/{ch}_1kb.txt'
    ]

    for chrom_file, name, n_read in zip(chrom_files, names, n_reads):
        count = 0
        for line in open(chrom_file):
            if count % 500000 == 0:
                print('{0} Line: {1}'.format(name, count))
            count += 1
            [p1, p2, v] = line.strip().split()
            p1, p2, v = int(p1) // res, int(p2) // res, float(v)
            if p1 > p2:
                p1, p2 = p2, p1
            if p2 - p1 >= n_strata:
                continue
            if p1 >= len(strata[p2 - p1]):
                continue
            strata[p2 - p1][p1] += v / n_read
    return strata


def generate_regions(cell_line, ch, strata, epigenetic_data, inp_window=15):
    # sr2human = np.load('sr2human.npy')
    if cell_line == 'mESC':
        st_ed_pos = MOUSE_CHR_SIZES
    else:
        st_ed_pos = HUMAN_CHR_SIZES
    st, ed = st_ed_pos[ch]

    for pos in range(st, ed - 499999, 50000):
        epi = epigenetic_data[ch][pos // 200 - (inp_window // 2): pos // 200 + 1250 + (inp_window // 2), :]
        epis = np.array([epi])

        hic = np.zeros((250, 250))
        for i in range(250):
            # for j in range(pos // 1000, pos // 1000 + 250 - i):
            #     ii, jj = i, i + j - pos // 1000
            #     hic[ii, jj] = strata[i][j]
            #     hic[jj, ii] = strata[i][j]
            for j in range(i, 250):
                hic[i, j] = strata[j - i][i + pos // 1000]
                hic[j, i] = strata[j - i][i + pos // 1000]
        # hic = hic * sr2human
        f = interp2d(np.arange(250), np.arange(250), hic)
        new_co = np.linspace(-0.4, 249.4, 1250)
        hic = f(new_co, new_co)
        hic = np.log(hic + 1)
        # np.save(f'outputs/pred_{ch}_{pos}_{pos + 250000}.npy', hic)
        hics = np.array([hic])
        yield pos, hics, epis


def cal_normalization_mat_200bp():
    m = np.ones((1250, 1250))
    m[:250, :250] += 1
    m[:500, :500] += 1
    m[:750, :750] += 1
    m[:1000, :1000] += 1
    m[250:, 250:] += 1
    m[500:, 500:] += 1
    m[750:, 750:] += 1
    m[1000:, 1000:] += 1
    return m


def load_epigenetic_data(chromosomes, epi_names, cell_type='HFF', verbose=1):
    epigenetic_data = {}
    res = 200

    for ch in chromosomes:
        epigenetic_data[ch] = None
        for i, k in enumerate(epi_names):
            path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi'
            path = f'{path}/{cell_type}/{ch}/{ch}_{res}bp_{k}.npy'
            s = np.load(path)
            # print(ch, k, s.shape)
            s = zscore(s)
            if verbose:
                print(ch, k, len(s))
            if i == 0:
                epigenetic_data[ch] = np.zeros((len(s), len(epi_names)))
            epigenetic_data[ch][:, i] = s
            # epigenetic_data[ch] = epigenetic_data[ch].T
    return epigenetic_data


def model_loop(first_layer=[96, 3],
               conv_layer_filters=[96, 96], conv_layer_windows=[3, 3],
               nBins=1250, nMarks=7, lr=0.0004, verbose=1):
    epi_data = Input(shape=(nBins + first_layer[1] - 1, nMarks))

    hidden_0 = Conv1D(first_layer[0], first_layer[1], activation='relu')(epi_data)

    if len(conv_layer_filters) > 0:
        hidden_c = [Conv1D(conv_layer_filters[0], conv_layer_windows[0], padding='same', activation='relu')(hidden_0)]
        for i in range(1, len(conv_layer_filters)):
            hidden_c.append(Conv1D(conv_layer_filters[i], conv_layer_windows[i],
                                   padding='same', activation='relu')(hidden_c[-1]))
    else:
        hidden_c = []

    combined = Concatenate(axis=-1)(hidden_c + [hidden_0])
    pred = Conv1D(400, 1)(combined)
    res = Dot(axes=(2, 2))([pred, pred])

    m = Model(inputs=epi_data, outputs=res)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
    return m


def args(nMarks=7):
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
        default=nMarks,
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


def normalize_strata(original_strata, _max, threshold=1.2, tol=1e-3):
    new_strata = deepcopy(original_strata) / np.mean(original_strata)
    iter = 0
    while np.max(new_strata) > threshold * _max + tol:
        iter += 1
        print(f'Iter {iter}, Current Max: {np.max(new_strata)}, Upper Bound: {threshold * _max}')
        new_strata[new_strata > threshold * _max] = threshold * _max
        new_strata = new_strata / np.mean(new_strata)
    return new_strata


def examples(chro='chr2', train='hESC', test='hESC', nMarks=6,
             path1='hESC7_temp_model_43.h5', path2='hESC7_loop_model_59.h5'):
    if nMarks == 3:
        epi_names = ['ATAC_seq', 'CTCF', 'H3K27ac']
    elif nMarks == 6:
        epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
                     'H3K27ac', 'H3K27me3']
    elif nMarks == 7:
        epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me2', 'H3K4me3',
                     'H3K27ac', 'H3K27me3']
    elif nMarks == 13:
        epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me2', 'H3K4me3',
                     'H3K9ac', 'H3K9me3', 'H3K27ac', 'H3K27me3', 'H3K36me3',
                     'H3K79me2', 'Rad21', 'Nanog']
    else:
        raise ValueError
    epi_hESC = load_epigenetic_data([chro], epi_names, cell_type=test)

    model1 = load_model(args(nMarks=nMarks), path1)
    model2 = model_loop(nMarks=nMarks)
    model2.load_weights(path2)

    strata = load_all_strata(chro)
    norm = cal_normalization_mat_200bp()
    lengths = load_chrom_sizes('hg38')
    lgth = lengths[chro] // 200 + 1
    new_strata = [np.zeros((lgth - i,)) for i in range(1000)]

    for pos, hics, epis in generate_regions('HFF', chro, strata, epi_hESC):
        print(pos)
        output1 = model1.predict([hics, epis]).reshape([1250, 1250])
        output2 = model2.predict(epis[:, 6:-6, :]).reshape([1250, 1250])
        for x1 in range(1250):
            for x2 in range(-4, 5):
                if 0 <= x1 + x2 < 1250:
                    output2[x1, x1 + x2] = 0
        output2[output2 < np.quantile(output2, 0.99)] = 0
        output2 = convolve2d(output2, np.ones((3, 3)) / 9, mode='same')
        output = output1 + output2 / 2
        output = np.exp(output) - 1
        output = output / norm

        for i in range(1000):
            strata_i = np.diag(output[i:, :1250 - i])
            # print(strata_i.shape, pos // 200 + 500 - i, pos // 200 + 750 - i, 500 - i // 2)
            new_strata[i][pos // 200: pos // 200 + 1250 - i] = strata_i

    pp = f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/MicroC/{test}'
    # new_strata_path = '/nfs/turbo/umms-drjieliu/usr/temp_Fan/sr_strata'
    micro_strata = []
    for i in range(len(new_strata)):
        print('Saving:', i)
        r = np.load(f'{pp}/{chro}/{chro}_200bp_strata_{i}.npy')
        r = r / np.mean(r)
        micro_strata.append(r)
        mx = np.max(r)
        new_strata[i] = normalize_strata(new_strata[i], mx)
        # np.save(f'{new_strata_path}/{train}_{test}_{ch}_strata_{i}.npy', new_strata[i])

    hicrep(test, new_strata, micro_strata, outputs=f'{train}_{test}_{nMarks}_{chro}_hicrep_h5.txt')


if __name__ == '__main__':
    combinations = [
        ['HFF', 'HFF', 6, '../HFF6/HFF6_temp_model_49.h5', '../HFF6/HFF6_loop_model_39.h5'],
    ]

    for train, test, nMarks, path1, path2 in combinations:
        for chro in ['chr2', 'chr4', 'chr17', 'chr18']:
            examples(chro, train, test, nMarks, path1, path2)
