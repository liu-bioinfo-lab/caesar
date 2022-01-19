import re
import os
from time import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from scipy.signal import convolve, convolve2d
from CAESAR_model import CAESAR

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
    rg_path = re.sub('CAESAR_review.*$', f'CAESAR_review/00_utils/{reference_genome}.chrom.sizes', my_path)
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


def load_epigenetic_data(cell_lines, chromosomes, epi_names, verbose=1):
    my_path = os.path.abspath(os.path.dirname(__file__))
    rg_path = re.sub('CAESAR_review.*$', f'CAESAR_review/00_utils/epi_norm_factors.csv', my_path)
    df = pd.read_csv(rg_path, sep='\t', index_col=0)

    epigenetic_data = {}
    res = 200
    for cell_line in cell_lines:
        epi_data = {}
        for ch in chromosomes:
            epi_data[ch] = None
            for i, k in enumerate(epi_names):
                norm_factor = df.loc[k, cell_line]
                path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data'
                src = '{0}/Epi/{1}/{2}/{3}_{4}bp_{5}.npy'.format(path, cell_line, ch, ch, res, k)
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


def strata2matrices(strata, ch_coord, resolution, gap=50000, size=250000, interp=False, thre=6, smooth=False):
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


def training_data_generator(ch_coord, hic_cell_lines, micro_cell_line, epi_names, pos_enc_dim=8,
                            n_epoches=100, batch_size=20):
    size = 1250
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        assert st % 1000 == 0
        assert ed % 1000 == 0

    # positional encoding
    _tm = time()
    pos_encoding = positional_encoding(size, pos_enc_dim)
    pos_encoding = np.array([pos_encoding for _ in range(batch_size)])
    # print(pos_encoding.shape)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating positional encoding...', '{}min:{}sec'.format(t_min, t_sec))

    # load epigenomic data
    epi_features = load_epigenetic_data([micro_cell_line], ch_coord.keys(), epi_names)
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
            path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/HiC/'
            strata = None
            for cell_line in hic_cell_lines:
                s = np.load(f'{path}/{cell_line}/{ch}/{ch}_1000bp_strata_{i}.npy')[st // 1000: ed // 1000 - i]
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
    # _tm = time()
    print(f'Finish processing {len(hic_mats)} HiC maps...', '{}min:{}sec'.format(t_min, t_sec))
    del hic_strata

    # Load Micro-C strata
    # hic_strata = {ch: [] for ch in ch_coord}
    # for ch in ch_coord:
    #     st, ed = ch_coord[ch]
    #     print(' Loading Micro-C strata:', ch, st, ed)
    #     for i in range(1250):
    #         if i % 125 == 0:
    #             print(' ', i, '/', 1250)
    #         path = '/nfs/turbo/umms-drjieliu/usr/temp_Fan/filtered_MicroC'
    #         strata = np.load(f'{path}/{micro_cell_line}/{ch}/{ch}_200bp_strata_{i}.npy')
    #         hic_strata[ch].append(strata[st // 200: ed // 200 - i] / np.mean(strata))
    # (t_min, t_sec) = divmod(time() - _tm, 60)
    # _tm = time()
    # print('Finish loading Micro-C strata...', '{}min:{}sec'.format(t_min, t_sec))

    # Convert to output contact maps
    # micro_mats = strata2matrices(hic_strata, ch_coord, resolution=200, interp=False, smooth=True)
    # (t_min, t_sec) = divmod(time() - _tm, 60)
    # print(f'Finish processing {len(micro_mats)} MicroC maps...', '{}min:{}sec'.format(t_min, t_sec))
    # del hic_strata

    # Initiating iteration:
    all_keys = list(hic_mats.keys())
    idx = list(range(len(all_keys)))
    # print(idx)
    _tm = time()
    # np.random.seed(0)

    print('Start training:')
    for _epoch in range(n_epoches):
        print('Epoch:', _epoch + 1)
        # np.random.shuffle(idx)

        for _batch in range(len(idx) // batch_size):
            if _epoch != 0 or _batch != 0:
                (t_min, t_sec) = divmod(time() - _tm, 60)
                print('{}min:{}sec'.format(t_min, t_sec))
            _tm = time()

            print(' Batch:', _batch + 1)
            batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]
            # print(batch_idx)
            kys = [all_keys[__] for __ in batch_idx]
            print(kys)

            hics = np.array(
                [hic_mats[all_keys[_id]] for _id in batch_idx]
            )

            # micros = np.array(
            #     [micro_mats[all_keys[_id]] for _id in batch_idx]
            # )

            epis = np.zeros((batch_size, 1250, len(epi_names)))
            for i in range(batch_size):
                _id = all_keys[batch_idx[i]]
                [ch, pos] = _id.split('_')
                pos = int(pos)
                epis[i, :, :] = \
                    epi_features[micro_cell_line][ch][pos // 200: pos // 200 + 1250, :]

            yield _epoch+1, _batch+1, (hics, epis, pos_encoding), kys


def load_normed_hic_strata(ch_coord, hic_cell_lines):
    _tm = time()
    hic_strata = {ch: [] for ch in ch_coord}
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        print(' Loading HiC strata:', ch, st, ed)
        for i in range(250):
            if i % 25 == 0:
                print(' ', i, '/', 250)
            path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/HiC/'
            strata = None
            for cell_line in hic_cell_lines:
                s = np.load(f'{path}/{cell_line}/{ch}/{ch}_1000bp_strata_{i}.npy')
                if strata is None:
                    strata = s
                else:
                    strata += s
            strata = strata / len(hic_cell_lines)
            hic_strata[ch].append(strata[st // 1000: ed // 1000 - i] / np.mean(strata))
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish loading HiC strata...', '{}min:{}sec'.format(t_min, t_sec))
    return hic_strata


def load_micro_strata(ch_coord, micro_cell_line):
    _tm = time()
    hic_strata = {ch: [] for ch in ch_coord}
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        print(' Loading Micro-C strata:', ch, st, ed)
        for i in range(1250):
            if i % 125 == 0:
                print(' ', i, '/', 1250)
            path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/MicroC/'
            strata = np.load(f'{path}/{micro_cell_line}/{ch}/{ch}_200bp_strata_{i}.npy')
            hic_strata[ch].append(strata[st // 200: ed // 200 - i] / np.mean(strata))
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish loading Micro-C strata...', '{}min:{}sec'.format(t_min, t_sec))
    return hic_strata


def correction_mat():
    m = np.ones((1250, 1250))
    m[:250, :250] += 1
    m[:500, :500] += 1
    m[:750, :750] += 1
    m[:1000, :1000] += 1
    m[250:, 250:] += 1
    m[500:, 500:] += 1
    m[750:, 750:] += 1
    m[1000:, 1000:] += 1
    return 1 / m


def update_strata(ch_coord, pred_strata, ky, mat, correction):
    m = mat * correction
    lst = ky.split('_')
    ch, pos = lst[0], int(lst[1])
    st, ed = ch_coord[ch]

    for i in range(1250):
        strata_i = np.diag(m[i:, :1250 - i])
        # print(strata_i.shape, pos // 200 + 500 - i, pos // 200 + 750 - i, 500 - i // 2)
        # print(i, pos, st, ed, len(pred_strata[ch][i]))
        pred_strata[ch][i][(pos - st) // 200: (pos - st) // 200 + 1250 - i] += strata_i
    return pred_strata


def scc(strata1, strata2, output, h=5):
    cors = np.zeros((1000,))
    window_size = 2 * h + 1
    for i in range(1000):
        print('Calculating:', i)
        pred = np.zeros((len(strata1[i]) - window_size + 1,))
        real = np.zeros((len(strata1[i]) - window_size + 1,))

        for j in range(i - window_size + 1, i + window_size):
            kernel_size = window_size - abs(i - j)
            kernel = np.ones((kernel_size,)) / (window_size ** 2)
            # kernel = np.ones((kernel_size,)) / kernel_size

            if j >= 1000:
                continue

            r_pred = convolve(strata1[abs(j)], kernel, mode='valid')
            r_real = convolve(strata2[abs(j)], kernel, mode='valid')

            delta_length = len(r_pred) - len(pred)
            if delta_length > 0:
                pred += r_pred[delta_length // 2: - delta_length // 2]
                real += r_real[delta_length // 2: - delta_length // 2]
            elif delta_length == 0:
                pred += r_pred
                real += r_real
            else:
                print('Impossible!')
                raise ValueError('Conv1D')

        cor = np.corrcoef(pred, real)[0, 1]
        cors[i] = cor
    np.savetxt(output, cors)


if __name__ == '__main__':
    my_path = os.path.abspath(os.path.dirname(__file__))
    model = CAESAR()
    model.load_weights(f'{my_path}/temp_model_100.h5')

    hic_cell_lines = ['HFF']
    micro_cell_line = 'HFF'

    # ch_coord = {'chr1': (17600000, 25500000)}
    ch_coord = HUMAN_CHR_SIZES
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    generator = training_data_generator(ch_coord, hic_cell_lines, micro_cell_line, epi_names,
                                        pos_enc_dim=8, n_epoches=1, batch_size=1)

    # hic_all_strata = load_normed_hic_strata(ch_coord, hic_cell_lines)
    micro_all_strata = load_micro_strata(ch_coord, micro_cell_line)
    pred_all_strata = {ch: [np.zeros(elm.shape) for elm in micro_all_strata[ch]] for ch in micro_all_strata}
    correction = correction_mat()

    for epoch, batch, (hics, epis, pos_enc), kys in generator:
        _res = model.predict([hics, epis, pos_enc])[0, :, :]
        _epi = epis[0, :, :].T
        ky = kys[0]

        pred_all_strata = update_strata(ch_coord, pred_all_strata, ky, _res, correction)

    path = '/nfs/turbo/umms-drjieliu/usr/temp_Fan/05_HFF_stripe_denoise_all'
    for ch in ch_coord:
        if not os.path.exists(f'{path}/{ch}'):
            os.mkdir(f'{path}/{ch}')
        for i in range(len(pred_all_strata[ch])):
            np.save(f'{path}/{ch}/strata_{i}.npy', pred_all_strata[ch][i])

        scc(pred_all_strata[ch], micro_all_strata[ch], f'{my_path}/{ch}_pred_micro.txt')
        # scc(pred_all_strata[ch], hic_all_strata[ch], f'{my_path}/{ch}_pred_hic.txt')
        # scc(micro_all_strata[ch], hic_all_strata[ch], f'{my_path}/{ch}_micro_hic.txt')




