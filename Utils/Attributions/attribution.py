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
import keras.backend as K


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


def load_micro_strata(cell_type='hESC', ch='chr2'):
    micro_strata = []
    pp = f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/MicroC/{cell_type}'
    for i in range(1000):
        print('Loading:', i)
        r = np.load(f'{pp}/{ch}/{ch}_200bp_strata_{i}.npy')
        micro_strata.append(r)
    return micro_strata


def strata2mat(strata, direction='v'):
    length = len(strata[0])
    mat = np.zeros((length, 1000))
    if direction == 'h':
        for i in range(1000):
            mat[:len(strata[i]), i] = strata[i]
    else:
        for i in range(1000):
            mat[i:i+len(strata[i]), i] = strata[i]
    return mat


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
        verbose=0
    )
    model.load_weights(weights)
    return model


def load_epigenetic_data(chromosomes, epi_names, cell_type='mESC', verbose=1):
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


def model_loop(weights, first_layer=[96, 3],
               conv_layer_filters=[96, 96], conv_layer_windows=[3, 3],
               nBins=1250, nMarks=6, lr=0.0004, verbose=0):
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
    m.load_weights(weights)
    return m


def int_gradient_strata(ch='chr2', steps=100):
    # Load epi
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    epi_data = load_epigenetic_data([ch], epi_names, cell_type='HFF')

    positions = [
        (10100000, 90, 120, 1000, 1030)
    ]

    path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'
    for (pos, x1, x2, y1, y2) in positions:
        print(pos)
        hic = np.load(f'{path}/HiC/mix/{ch}/{ch}_1000bp_{pos}_{pos + 250000}.npy')

        model1 = load_model(args(), 'HFF6_temp_model_49.h5')
        model2 = model_loop('HFF6_loop_model_39.h5')
        epi = epi_data[ch][pos // 200 - 7: pos // 200 + 1257, :]
        # np.save(f'epi_{pos}.npy', epi_data[ch][pos // 200: pos // 200 + 1250, :].T)

        grad1 = K.gradients(
            model1.outputs[0][:, x1:x2, y1:y2],
            model1.inputs)
        grad2 = K.gradients(
            model2.outputs[0][:, x1:x2, y1:y2],
            model2.input)
        sess = K.get_session()

        attr = np.zeros((steps, 1250, 6))
        for j in range(steps // 20):
            hics = np.zeros((20, 1250, 1250))
            for k in range(20):
                hics[k, :, :] = hic
            epis = np.zeros((20, 1264, 6))
            for k in range(20):
                epis[k, :, :] = (k + j * 20) / 100 * epi
            grad_res1 = sess.run(
                grad1,
                feed_dict={
                    model1.inputs[0]: hics,
                    model1.inputs[1]: epis
                })[1][:, 7:-7, :]
            grad_res2 = sess.run(
                grad2,
                feed_dict={
                    model2.input: epis[:, 6:-6, :]
                })[0][:, 1:-1, :]
            attr[j * 20:j*20 + 20, :, :] = grad_res1 + grad_res2 * 0.5
        K.clear_session()

        attr = np.sum(attr, axis=0) / steps
        np.save(f'att_{pos}.npy', attr.T)


int_gradient_strata()


