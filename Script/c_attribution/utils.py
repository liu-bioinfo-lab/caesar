from keras.layers import Input, Dropout, Concatenate, Permute, Conv1D, Add, Dot, Multiply
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from tensorflow import Graph, Session
from layers import GraphConv
import numpy as np
from scipy.stats import zscore
from scipy.interpolate import interp2d
import os


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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mouse_start_end = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                   'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                   'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                   'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                   'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                   'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                   'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)}
human_start_end = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }


def parse_coordinate(coordinate):
    """
    Args:
        coordinate (str):
    Example:
        >>> parse_coordinate('chr1:153500000-153501000, chr1:153540000-153542000')
        ['chr1', 153500000, 153501000, 153540000, 153542000]
    Return:
         A list [chromosome] + [coordinate of four corners]
    """

    try:
        pos1, pos2 = [elm.strip() for elm in coordinate.split(',')]
        pos1 = pos1.replace(':', '-')
        c1, p11, p12 = [elm.strip() for elm in pos1.split('-')]
        pos2 = pos2.replace(':', '-')
        c2, p21, p22 = [elm.strip() for elm in pos2.split('-')]
        p11, p12, p21, p22 = int(p11), int(p12), int(p21), int(p22)
    except:
        raise ValueError('Invalid coordinate string!')

    if c1 != c2:
        raise ValueError('Intrachromosomal contacts only!')

    if p22 - p11 > 200000:
        raise ValueError('Short-distance contacts (within 200 kb) only!')

    return [c1, p11, p12, p21, p22]


def find_250kb_region(position):
    """Find a 200-kb region which covers the chosen position best
    For example, for contacts between chr1:153500000-153501000, chr1:153500000-153501000,
    region chr1:153400000-153600000 is the best.
    Then, change the original p11, p22, p21, p22 into the coordinate in the 200-kb region
    (Since the resolution is 200 bp, the range will be between 0-999)
    Args:
        position (list):
    Example:
        >>> find_250kb_region(['chr1', 153500000, 153501000, 153540000, 153542000])
        ['chr1', 153400000, 500, 505, 700, 710]
    Return:
         A list [chromosome, region_start_position] + [new coordinates in this sub-region]
    """
    human_start = human_start_end[position[0]][0]
    resolution = 200
    p11, p22 = position[1], position[4]
    center = (p11 + p22) / 2
    closest_center = int(round((center - human_start) / 125000) * 125000 + human_start)
    start_pos = closest_center - 125000
    new_pos = [int(round((elm - start_pos) / resolution)) for elm in position[1:]]
    return [position[0], start_pos] + new_pos


def load_all_data(cell_line, ch, start_pos, signals, hic_path, hic_resolution, epi_path):
    hic = load_hic_data(cell_line, ch, start_pos, hic_path, hic_resolution)
    epi = load_epigenetic_data(cell_line, [ch], signals, epi_path)
    epi = epi[ch][start_pos // 200 - 7: start_pos // 200 + 1257, :]
    return hic, epi


def normalize_HiC(hic, observed_exps):
    exps = np.loadtxt('HiC_exps.txt')
    for i in range(len(hic)):
        for j in range(len(hic)):
            hic[i, j] = hic[i, j] / observed_exps[abs(i - j)] * exps[abs(i - j)]
    return hic


def load_hic_data(cell_line, ch, pos, hic_file, hic_resolution):
    resolution = 200
    dim = 1250
    length = load_chrom_sizes('mm10')[ch] if cell_line == 'mESC' else load_chrom_sizes('hg38')[ch]
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
    return hic


def load_epigenetic_data(cell_line, chromosomes, signals, epi_path):
    functional_data = {}

    for chrom in chromosomes:
        functional_data[chrom] = None
        for i, k in enumerate(signals): # body_of_pancreas_m37_chr4_200bp_H3K27ac.npy
            # s = np.load(f'{source_path}/source_data/pancreas_{chrom}_{k}_200bp.npy')
            s = np.load(f'{epi_path}/{cell_line}/{chrom}/{chrom}_200bp_{k}.npy')
            s = zscore(s)
            if i == 0:
                functional_data[chrom] = s
            else:
                functional_data[chrom] = np.vstack((functional_data[chrom], s))
        functional_data[chrom] = functional_data[chrom].T
    return functional_data


def model_fn(model_weights='HFF6_temp_model_39.h5',
             first_layer=[96, 15], gcn_layers=[96, 96],
             conv_layer_filters=[96], conv_layer_windows=[15],
             nBins=1250, nMarks=6, lr=0.0001, verbose=False):  # nMarks was 8
    hic = Input(shape=(nBins, nBins))
    epi_data = Input(shape=(nBins + first_layer[1] - 1, nMarks))

    hidden_0 = Conv1D(first_layer[0], first_layer[1], activation='relu')(epi_data)

    hidden_g = [GraphConv(gcn_layers[0], activation='relu')([hidden_0, hic])]
    for i in range(1, len(gcn_layers)):
        hidden_g.append(GraphConv(gcn_layers[i], activation='relu')([hidden_g[-1], hic]))

    hidden_c = [Conv1D(conv_layer_filters[0], conv_layer_windows[0], padding='same', activation='relu')(hidden_0)]
    for i in range(1, len(conv_layer_filters)):
        hidden_c.append(Conv1D(conv_layer_filters[i], conv_layer_windows[i],
                               padding='same', activation='relu')(hidden_c[-1]))

    combined = Concatenate(axis=-1)(hidden_g + hidden_c + [hic])
    pred = Conv1D(nBins, 1, activation='relu')(combined)
    pred_T = Permute([2, 1])(pred)
    res = Add()([pred, pred_T])

    m = Model(inputs=[hic, epi_data], outputs=res)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
    m.load_weights(model_weights)
    return m


def model_loop(model_weights='HFF6_loop_model_39.h5',
               first_layer=[96, 3], gcn_layers=[96, 96],
               conv_layer_filters=[96], conv_layer_windows=[3],
               nBins=1250, nMarks=6, lr=0.0001, verbose=1):
    hic = Input(shape=(nBins, nBins))
    epi_data = Input(shape=(nBins + first_layer[1] - 1, nMarks))
    mask = Input(shape=(nBins, nBins))

    hidden_0 = Conv1D(first_layer[0], first_layer[1], activation='relu')(epi_data)

    if len(gcn_layers) > 0:
        hidden_g = [GraphConv(gcn_layers[0], activation='relu')([hidden_0, hic])]
        for i in range(1, len(gcn_layers)):
            hidden_g.append(GraphConv(gcn_layers[i], activation='relu')([hidden_g[-1], hic]))
    else:
        hidden_g = []

    if len(conv_layer_filters) > 0:
        hidden_c = [Conv1D(conv_layer_filters[0], conv_layer_windows[0], activation='relu', padding='same')(hidden_0)]
        for i in range(1, len(conv_layer_filters)):
            hidden_c.append(Conv1D(conv_layer_filters[i], conv_layer_windows[i],
                                   padding='same', activation='relu')(hidden_c[-1]))
    else:
        hidden_c = []

    combined = Concatenate(axis=-1)(hidden_g + hidden_c + [hidden_0])
    pred = Conv1D(400, 1)(combined)
    res = Dot(axes=(2, 2))([pred, pred])
    res = Multiply()([res, mask])

    m = Model(inputs=[hic, epi_data, mask], outputs=res)
    m.compile(optimizer=Adam(lr=lr), loss='mse')

    if verbose:
        m.summary()
    m.load_weights(model_weights)
    return m


def int_grad(hic, epigenetic, positions, steps=100,
             model1_path='contact_profile_model_49.h5', model2_path='loop_model_45.h6'):
    functionals = np.zeros((steps, 1264, epigenetic.shape[1]))
    hics = np.zeros((steps, 1250, 1250))
    mask = np.zeros((steps, 1250, 1250))
    limit = 20
    msk = np.ones((1250, 1250))
    for i in range(1250):
        for j in range(max(0, i - limit), min(1250, i + limit + 1)):
            msk[i, j] = 0

    for i in range(steps):
        functionals[i, :, :] = epigenetic * i / steps
        hics[i, :, :] = hic
        mask[i, :, :] = msk
    # print('Input Data Loaded!')

    grad_res = np.zeros((steps, 1250, epigenetic.shape[1]))

    graph = Graph()
    with graph.as_default():
        with Session() as sess:
            model = model_fn(model1_path)
            model_lp = model_loop(model2_path)   # the path of the loop model

            grad = K.gradients(
                model.outputs[0][:, positions[0]:positions[1], positions[2]:positions[3]],
                model.inputs)

            grad_lp = K.gradients(
                model_lp.outputs[0][:, positions[0]:positions[1], positions[2]:positions[3]],
                model_lp.input)

            for s in range(0, steps, 20):
                _grad = sess.run(
                    grad,
                    feed_dict={
                        model.inputs[0]: hics[s:s+20, :, :],
                        model.inputs[1]: functionals[s:s+20, :, :]
                    }
                )[1][:, 7:-7, :]
                _grad_lp = sess.run(
                    grad_lp,
                    feed_dict={
                        model_lp.input[0]: hics[s:s+20, :, :],
                        model_lp.input[1]: hics[s:s+20, 6:-6, :],
                        model_lp.input[2]: mask[s:s+20, :, :]
                    }
                )[0][:, 1:-1, :]

                grad_res[s:s+20, :, :] = _grad + _grad_lp * 0.5
    grad_res = np.sum(grad_res, axis=0) / steps
    return grad_res.T  # Remove the 14kb padding region


