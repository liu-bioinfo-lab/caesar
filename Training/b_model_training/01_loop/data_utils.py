import os
from time import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from scipy.signal import convolve2d


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
    rg_path = f'{my_path}/{reference_genome}.chrom.sizes'
    f = open(rg_path)
    lengths = {}
    for line in f:
        [ch, l] = line.strip().split()
        lengths[ch] = int(l)
    return lengths


def positional_encoding(length=1250, position_dim=8):
    assert position_dim % 2 == 0
    position = np.zeros((length, position_dim))
    for i in range(position_dim // 2):
        position[:, 2 * i] = np.sin([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
        position[:, 2 * i + 1] = np.cos([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
    return position


def load_epigenetic_data(cell_lines, chromosomes, epi_names, processed_path, verbose=1):
    rg_path = '../../a_data_processing/01_epi_stats/epi_norm_factors.csv'
    df = pd.read_csv(rg_path, sep='\t', index_col=0)

    epigenetic_data = {}
    res = 200
    for cell_line in cell_lines:
        epi_data = {}
        for ch in chromosomes:
            epi_data[ch] = None
            for i, k in enumerate(epi_names):
                norm_factor = df.loc[k, cell_line]
                src = '{0}/Epi/{1}/{2}/{3}_{4}bp_{5}.npy'.format(processed_path, cell_line, ch, ch, res, k)
                s = np.load(src) * norm_factor
                s = np.log(s + 1)
                s[s > 4] = 4
                if verbose:
                    print(' Loading epigenomics...', ch, k, len(s))
                if i == 0:
                    epi_data[ch] = np.zeros((len(s), len(epi_names)))
                epi_data[ch][:, i] = s
        epigenetic_data[cell_line] = epi_data
    return epigenetic_data


def strata2matrices(strata, ch_coord, resolution, gap=125000, size=250000, interp=False, thre=6, smooth=False):
    matrices = {}
    id_size = size // resolution
    for ch in ch_coord:
        print(' Converting to matrices:', ch)
        st, ed = ch_coord[ch]
        id_gap = gap // resolution
        for i, st_pos in enumerate(range(st, ed - size + 1, gap)):
            ky = f'{ch}_{st_pos}'
            curr_id = i * id_gap
            mat = np.zeros((id_size, id_size))
            for j in range(id_size):
                # method 2
                slc = strata[ch][j][curr_id: curr_id + id_size - j]
                for k in range(len(slc)):
                    mat[k, k + j] = slc[k]
                    if j != 0:
                        mat[k + j, k] = slc[k]

            if interp:
                fold_change = 5
                fc_ = 1 / fold_change
                f = interp2d(np.arange(id_size), np.arange(id_size), mat)
                new_co = np.linspace(-0.5 + fc_ / 2, id_size - 0.5 - fc_ / 2, id_size * fold_change)
                mat = f(new_co, new_co)

            mat = np.log(mat + 1)
            if smooth:
                mat = convolve2d(mat, np.ones((3, 3)) / 9, mode='same')

            mat[mat > thre] = thre
            matrices[ky] = mat
    return matrices


def load_all_loops(cell_line):
    def filter_35():
        mat = np.zeros((35, 35))
        for t in range(16):
            mat[t:35 - t, t:35 - t] = t / 15
        return mat

    def filter_55():
        mat = np.zeros((55, 55))
        for t in range(26):
            mat[t:55 - t, t:55 - t] = t / 25
        return mat

    my_path = os.path.abspath(os.path.dirname(__file__))
    lp_path1 = f'{my_path}/micro_loops/{cell_line}_micro_loops_all_1kb/merged_loops.bedpe'
    lp_path2 = f'{my_path}/micro_loops/{cell_line}_micro_loops_all_5kb/merged_loops.bedpe'
    loops = {}
    for line in open(lp_path1):
        if not line.startswith('#'):
            lst = line.strip().split()[:6]
            ch = 'chr' + lst[0]
            if ch not in loops:
                loops[ch] = []
            loops[ch].append((int(lst[1]), int(lst[2]), int(lst[4]), int(lst[5])))
    for line in open(lp_path2):
        if not line.startswith('#'):
            lst = line.strip().split()[:6]
            ch = 'chr' + lst[0]
            if ch not in loops:
                loops[ch] = []
            loops[ch].append((int(lst[1]), int(lst[2]), int(lst[4]), int(lst[5])))
    for ch in loops:
        print(ch, len(loops[ch]))
        loops[ch] = sorted(loops[ch])
    return loops, filter_35(), filter_55()


def generate_mask(size=1250, distance=10, batch_size=20):
    msk = np.ones((size, size))
    for i in range(size):
        for j in range(max(0, i - distance), min(size, i + distance + 1)):
            msk[i, j] = 0
    batch_msk = np.array([msk for _ in range(batch_size)])
    return batch_msk


def training_data_generator(ch_coord, hic_cell_lines, micro_cell_line, epi_names, processed_path,
                            pos_enc_dim=8, n_epoches=100, batch_size=20):
    size = 1250
    max_distance = 250000
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        assert st % 1000 == 0
        assert ed % 1000 == 0
    mask = generate_mask(size=size, distance=10, batch_size=batch_size)

    # positional encoding
    _tm = time()
    pos_encoding = positional_encoding(size, pos_enc_dim)
    pos_encoding = np.array([pos_encoding for _ in range(batch_size)])
    # print(pos_encoding.shape)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating positional encoding...', '{}min:{}sec'.format(t_min, t_sec))

    # load epigenomic data
    epi_features = load_epigenetic_data([micro_cell_line], ch_coord.keys(), epi_names, processed_path)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating epigenomic features...', '{}min:{}sec'.format(t_min, t_sec))

    # Load Hi-C strata
    hic_strata = {ch: [] for ch in ch_coord}
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        print(' Loading HiC strata:', ch, st, ed)
        for i in range(250):
            if i % 25 == 0:
                print(' ', i, '/', 250)
            strata = None
            for cell_line in hic_cell_lines:
                s = np.load(f'{processed_path}/HiC/{cell_line}/{ch}/{ch}_1000bp_strata_{i}.npy')[st // 1000: ed // 1000 - i]
                if strata is None:
                    strata = s
                else:
                    strata += s
            strata = strata / len(hic_cell_lines)
            hic_strata[ch].append(strata)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish loading HiC strata...', '{}min:{}sec'.format(t_min, t_sec))

    # Convert to input contact maps
    hic_mats = strata2matrices(hic_strata, ch_coord, resolution=1000, interp=True, smooth=False)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print(f'Finish processing {len(hic_mats)} HiC maps...', '{}min:{}sec'.format(t_min, t_sec))
    del hic_strata

    # Load Micro-C strata
    hic_strata = {ch: [] for ch in ch_coord}
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        print(' Loading Micro-C strata:', ch, st, ed)
        for i in range(1250):
            if i % 125 == 0:
                print(' ', i, '/', 1250)
            strata = np.load(f'{processed_path}/MicroC/{micro_cell_line}/{ch}/{ch}_200bp_strata_{i}.npy')
            hic_strata[ch].append(strata[st // 200: ed // 200 - i] / np.mean(strata))
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish loading Micro-C strata...', '{}min:{}sec'.format(t_min, t_sec))

    # Convert to output contact maps
    micro_mats = strata2matrices(hic_strata, ch_coord, resolution=200, interp=False, smooth=True)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print(f'Finish processing {len(micro_mats)} MicroC maps...', '{}min:{}sec'.format(t_min, t_sec))
    del hic_strata

    # loading loops
    loops, filter_35, filter_55 = load_all_loops(micro_cell_line)
    filter_m = {35: filter_35, 55: filter_55}
    (t_min, t_sec) = divmod(time() - _tm, 60)
    print(f'Finish loading {micro_cell_line} loops...', '{}min:{}sec'.format(t_min, t_sec))

    # Initiating iteration:
    all_keys = list(hic_mats.keys())
    idx = list(range(len(all_keys)))
    # print(idx)
    _tm = time()
    np.random.seed(0)

    print('Start training:')
    for _epoch in range(n_epoches):
        print('Epoch:', _epoch)
        np.random.shuffle(idx)

        for _batch in range(len(idx) // batch_size):
            if _epoch != 0 or _batch != 0:
                (t_min, t_sec) = divmod(time() - _tm, 60)
                print('{}min:{}sec'.format(t_min, t_sec))
            _tm = time()

            print(' Batch:', _batch)
            batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]
            # print(batch_idx)
            # keys = [all_keys[__] for __ in batch_idx]
            # print([all_keys[__] for __ in batch_idx])

            hics = np.array(
                [hic_mats[all_keys[_id]] for _id in batch_idx]
            )

            micros = np.array(
                [micro_mats[all_keys[_id]] for _id in batch_idx]
            )

            # process micros for loops
            micro_filter = np.zeros(micros.shape)
            for i in range(batch_size):
                _id = all_keys[batch_idx[i]]
                # print(_id, end=' ')
                [ch, pos] = _id.split('_')
                pos = int(pos)
                for loop in loops[ch]:
                    if pos > max(loop):
                        continue
                    elif pos + max_distance < min(loop):
                        break
                    else:
                        # print(loop, end=' ')
                        x1, x2, y1, y2 = loop
                        x1, x2, y1, y2 = (x1 - pos) // 200 - 15, (x2 - pos) // 200 + 15, \
                                         (y1 - pos) // 200 - 15, (y2 - pos) // 200 + 15
                        if x2 - x1 != y2 - y1:
                            continue
                        if min(x1, x2, y1, y2) >= 0 and max(x1, x2, y1, y2) < size:
                            micro_filter[i, x1:x2, y1:y2] = filter_m[x2 - x1]
                            micro_filter[i, y1:y2, x1:x2] = filter_m[x2 - x1]
                # print(' ')
            micros = micros * micro_filter

            epis = np.zeros((batch_size, 1250, len(epi_names)))
            for i in range(batch_size):
                _id = all_keys[batch_idx[i]]
                [ch, pos] = _id.split('_')
                pos = int(pos)
                epis[i, :, :] = \
                    epi_features[micro_cell_line][ch][pos // 200: pos // 200 + 1250, :]

            yield _epoch+1, _batch+1, (hics, epis, pos_encoding, mask), micros


