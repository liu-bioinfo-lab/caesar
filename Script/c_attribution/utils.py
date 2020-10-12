from keras.layers import Input, Dropout, Concatenate, Permute, Conv1D, Add, Dot
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from tensorflow import Graph, Session
from hirespy.attr.layers import GraphConv
import numpy as np
from scipy.stats import zscore
from scipy.interpolate import interp2d
import os
import seaborn as sns
from copy import deepcopy
import hirespy


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
# Util path
path = hirespy.app.config['TOOLS_FOLDER']
temp_folder = '/nfs/turbo/umms-drjieliu/usr/yyao/temp'
marks_folder = '/nfs/turbo/umms-drjieliu/usr/yyao/marks'


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


def find_1kb_region(position):
    """Find a 200-kb region which covers the chosen position best
    For example, for contacts between chr1:153500000-153501000, chr1:153500000-153501000,
    region chr1:153400000-153600000 is the best.
    Then, change the original p11, p22, p21, p22 into the coordinate in the 200-kb region
    (Since the resolution is 200 bp, the range will be between 0-999)
    Args:
        position (list):
    Example:
        >>> find_1kb_region(['chr1', 153500000, 153501000, 153540000, 153542000])
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


def load_all_data(ch, start_pos, signals, req):
    hic = load_hic_data(ch, start_pos, req)
    epi = load_epigenetic_data([ch], signals, req['marks'])
    epi = epi[ch][start_pos // 200 - 7: start_pos // 200 + 1257, :]
    return hic, epi


def load_hic_data(ch, start_pos, req):
    port = req['port']
    os.system(f'touch {temp_folder}/temp_{port}.txt')
    os.system(f'java -jar {path}/juicer_tools.jar dump observed NONE {req["tissue_path"]} \
{ch}:{start_pos}:{start_pos+250000} {ch}:{start_pos}:{start_pos+250000} BP 1000 {temp_folder}/temp_{port}.txt')
    mat = np.zeros((250, 250))
    # print(f'java -jar {path}/juicer_tools.jar dump observed NONE /var/www/data/main/H1-hESC_full.hic \
# {ch}:{start_pos}:{start_pos+250000} {ch}:{start_pos}:{start_pos+250000} BP 1000 {temp_folder}/temp.txt')
    for line in open(f'{temp_folder}/temp_{port}.txt'):
        p1, p2, v = line.strip().split()
        p1, p2, v = (int(p1) - start_pos) // 1000, (int(p2) - start_pos) // 1000, float(v)
        if p1 >= 250 or p2 >= 250:
            continue
        mat[p1, p2] += v
        if p1 != p2:
            mat[p2, p1] += v
    # print('MAT', mat.shape)
    f = interp2d(np.arange(250), np.arange(250), mat)
    new_co = np.linspace(-0.4, 249.4, 1250)
    mat = f(new_co, new_co)
    os.system(f'rm {temp_folder}/temp_{port}.txt')
    return np.log(mat + 1)


def load_epigenetic_data(chromosomes, signals, marks):
    """Load all required epigenetic data and normalize
    Args:
        chromosomes (list): chromosomes to calculate
        signals (list): epigenetic datasets
    Return:
        A dict {chromosome: numpy.array (500 * num_of_functional_datasets)}
    """
    functional_data = {}

    for chrom in chromosomes:
        functional_data[chrom] = None
        for i, k in enumerate(signals): # body_of_pancreas_m37_chr4_200bp_H3K27ac.npy
            # s = np.load(f'{source_path}/source_data/pancreas_{chrom}_{k}_200bp.npy')
            s = np.load(f'{marks_folder}/{marks}/{marks}_{chrom}_200bp_{k}.npy')
            s = zscore(s)
            # print('Loading:', chrom, k, len(s))
            if i == 0:
                functional_data[chrom] = s
            else:
                functional_data[chrom] = np.vstack((functional_data[chrom], s))
        functional_data[chrom] = functional_data[chrom].T
        # print(functional_data[chrom].shape)
    return functional_data


def model_fn(model_weights='HFF6_temp_model_39.h5',
             first_layer=[96, 15], gcn_layers=[96, 96],
             conv_layer_filters=[96], conv_layer_windows=[15],
             nBins=1250, nMarks=6, lr=0.0001, verbose=False):  # nMarks was 8
    """
    Build the GCN model
    Args:
        first_layer (list): n_filters and n_windows for the layer after input
        gcn_layers (list): n_filters for the following GCN layers
        conv_layer_filters (list): n_filters for the Conv1D layers
        conv_layer_windows (list): n_windows for the Conv1D layers
        nBins (int): size of input matrices
        nMarks (int):
        verbose (int):
    Return:
         model (keras.Model.model)
    """

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
    m.load_weights(os.path.join(path, model_weights))
    return m


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


# def build_model(epigenetic_datasets, parameter_path='model_v1.12.h5', verbose=1):
#     """Build the model and load the parameters
#
#     Args:
#         epigenetic_datasets (list): epigenetic datasets to use
#         parameter_path (str):
#         verbose (int): whether to print
#
#     Return:
#          A model (keras.models.Model)
#     """
#     func = Input(shape=(1264, len(epigenetic_datasets)))
#     sm_func = Conv1D(96, 15, activation='relu')(func)  # 2500 * 32
#     sm_func = BatchNormalization()(sm_func)
#
#     hic = Input(shape=(1250, 1250))
#
#     func_1 = GraphConv(128, activation='relu')([sm_func, hic])
#     func_2 = GraphConv(128, activation='relu')([func_1, hic])
#     func_3 = GraphConv(96, activation='relu')([func_2, hic])
#
#     func_5 = Conv1D(96, 5, padding='same', activation='relu')(sm_func)
#
#     all_func = Concatenate(axis=-1)([func_1, func_2, func_3, func_5])  # 2500 * 320
#
#     combined = Concatenate(axis=-1)([all_func, hic])
#     combined = Dropout(rate=0.15)(combined)
#
#     pred = Conv1D(1000, 1, activation='relu')(combined)
#     pred_T = Permute([2, 1])(pred)
#     res = Add()([pred, pred_T])
#
#     m = Model(inputs=[func, hic], outputs=res)
#     m.compile(optimizer='adam', loss='mse')
#     if verbose:
#         m.summary()
#     m.load_weights(os.path.join(path, parameter_path))
#     return m


def int_grad(hic, epigenetic, positions, steps=100):
    """
    :param chrom:
    :param start_pos:
    :param slice_start_pos:
    :param slice_end_pos:
    :param time_steps: (When calculating integral, we use the summation of several time steps to replace.)
    :return:
    """
    functionals = np.zeros((steps, 1264, epigenetic.shape[1]))
    hics = np.zeros((steps, 1250, 1250))
    for i in range(steps):
        functionals[i, :, :] = epigenetic * i / steps
        hics[i, :, :] = hic
    # print('Input Data Loaded!')

    ###############################################################
    grad_res = np.zeros((steps, 1250, epigenetic.shape[1]))
    ###############################################################

    graph = Graph()
    with graph.as_default():
        with Session() as sess:
            model = model_fn()
            #######################################################################
            model_lp = model_loop(weights)   # the path of the loop model
            ########################################################################

            grad = K.gradients(
                model.outputs[0][:, positions[0]:positions[1], positions[2]:positions[3]],
                model.inputs)

            #########################################################################
            grad_lp = K.gradients(
                model_lp.outputs[0][:, positions[0]:positions[1], positions[2]:positions[3]],
                model_lp.input)
            #########################################################################

            for s in range(0, steps, 20):
                _grad = sess.run(
                    grad,
                    feed_dict={
                        model.inputs[0]: hics[s:s+20, :, :],
                        model.inputs[1]: functionals[s:s+20, :, :]
                    }
                )[1][:, 7:-7, :]  ######################################################

                #######################################################################
                _grad_lp = sess.run(
                    grad_lp,
                    feed_dict={
                        model_lp.input: functionals[s:s+20, 6:-6, :]
                    }
                )[0][:, 1:-1, :]
                #######################################################################

                grad_res[s:s+20, :, :] = _grad + _grad_lp * 0.4 ##################################
            K.clear_session()  ####################################
    grad_res = np.sum(grad_res, axis=0) / steps
    return grad_res  # Remove the 14kb padding region


def save_bigBed(attributions, signals, ch, start_pos, req):
    port = req['port']
    # original_attribution = deepcopy(attributions)
    # Set threshold:
    threshold = np.quantile(attributions, 0.999)
    # Convert into discrete values
    attributions = attributions / threshold * 1000  # 500 + 500
    attributions[np.where(attributions >= 1000)] = 999
    attributions[np.where(attributions < -1000)] = -999
    #attributions[np.where(attributions < 0)] = 0
    # Load palette
    # plt = sns.color_palette('coolwarm', n_colors=1000)
    # plt = np.array([[int(round(255 * val)) for val in line] for line in plt], dtype=int)

    # Save bed file
    for i, signal in enumerate(signals):
        f = open(f'{temp_folder}/{signal}_temp_{port}.bed', 'w+')
        # f.write(f'track name="attribution_{i}" description="Item RGB demonstration" itemRgb="On" ')
        s = attributions[:, i]

        for j in range(len(s)):
            if s[j] < 0.5:
                continue
            # rgb = plt[int(attributions[j, i])]
            # f.write(f'{ch} {start_pos + j * 200} {start_pos + j * 200 + 200} peak {s[j]} + {start_pos + j * 200} {start_pos + j * 200 + 200} {rgb[0]},{rgb[1]},{rgb[2]}\n')
            # f.write(f'{ch} {start_pos + j * 200} {start_pos + j * 200 + 200} . {int(round(s[j]))}\n')
            f.write(f'{ch} {start_pos + j * 200} {start_pos + j * 200 + 200} {int(round(s[j]))}\n')

        for c in human_start_end:
            # if c != ch:
            f.write(f'{c} 0 200 500\n')

        f.close()

        # Convert into bigBed
        os.system(f'sort -k1,1 -k2,2n {temp_folder}/{signal}_temp_{port}.bed > {temp_folder}/{signal}_sorted_temp_{port}.bed')
        # os.system(f'{path}/bedToBigBed {source_path}/temp/{signal}_sorted_temp.bed {path}/hg38.chrom.sizes {source_path}/output_bigBed/{signal}_temp.bigBed')
        os.system(f'{path}/bedGraphToBigWig {temp_folder}/{signal}_sorted_temp_{port}.bed {path}/hg38.chrom.sizes {temp_folder}/output_bigWig/{signal}_temp_{port}.bigWig')
        os.system(f'rm {temp_folder}/{signal}_temp_{port}.bed')
        #os.system(f'rm {source_path}/temp/{signal}_sorted_temp.bed')
        os.system(f'mv {temp_folder}/output_bigWig/{signal}_temp_{port}.bigWig {req["tracks_path"]}/{signal}.bigWig')
