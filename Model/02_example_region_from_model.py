import os
from time import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from scipy.signal import convolve2d
from CAESAR_model import CAESAR, CAESAR_loop


def positional_encoding(length=1250, position_dim=8):
    assert position_dim % 2 == 0
    position = np.zeros((length, position_dim))
    for i in range(position_dim // 2):
        position[:, 2 * i] = np.sin([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
        position[:, 2 * i + 1] = np.cos([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
    return position


def load_epigenetic_data(cell_lines, chromosomes, epi_names, processed_path, verbose=1):
    # rg_path = '../../a_data_processing/01_epi_stats/epi_norm_factors.csv'
    my_path = os.path.abspath(os.path.dirname(__file__))
    rg_path = f'{my_path}/utils/epi_norm_factors_DNase.csv'
    df = pd.read_csv(rg_path, sep='\t', index_col=0)

    epigenetic_data = {}
    res = 200
    for cell_line in cell_lines:
        epi_data = {}
        for ch in chromosomes:
            epi_data[ch] = None
            for i, k in enumerate(epi_names):
                norm_factor = df.loc[k, cell_line]
                src = '{0}/{1}/{2}_{3}bp_{4}.npy'.format(processed_path, ch, ch, res, k)
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


def generate_mask(size=1250, distance=10, batch_size=20):
    msk = np.ones((size, size))
    for i in range(size):
        for j in range(max(0, i - distance), min(size, i + distance + 1)):
            msk[i, j] = 0
    batch_msk = np.array([msk for _ in range(batch_size)])
    return batch_msk


def training_data_generator(ch_coord, hic_cell_lines, micro_cell_line, epi_names, processed_path,
                            pos_enc_dim=8, n_epoches=1, batch_size=1):
    size = 1250
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
                s = np.load(f'{processed_path}/input_hic/{cell_line}/{ch}/{ch}_1000bp_strata_{i}.npy')[st // 1000: ed // 1000 - i]
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

    # Initiating iteration:
    all_keys = list(hic_mats.keys())
    idx = list(range(len(all_keys)))
    # print(idx)
    _tm = time()

    print('Start training:')
    for _epoch in range(n_epoches):
        print('Epoch:', _epoch)
        for _batch in range(len(idx) // batch_size):
            if _epoch != 0 or _batch != 0:
                (t_min, t_sec) = divmod(time() - _tm, 60)
                print('{}min:{}sec'.format(t_min, t_sec))
            _tm = time()

            print(' Batch:', _batch)
            batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]
            kys = [all_keys[__] for __ in batch_idx]

            hics = np.array(
                [hic_mats[all_keys[_id]] for _id in batch_idx]
            )

            epis = np.zeros((batch_size, 1250, len(epi_names)))
            for i in range(batch_size):
                _id = all_keys[batch_idx[i]]
                [ch, pos] = _id.split('_')
                pos = int(pos)
                epis[i, :, :] = \
                    epi_features[micro_cell_line][ch][pos // 200: pos // 200 + 1250, :]

            yield _epoch+1, _batch+1, (hics, epis, pos_encoding, mask), kys


if __name__ == '__main__':
    # Get current path
    my_path = os.path.abspath(os.path.dirname(__file__))

    #######################################################################################
    #  The below lines need to be consistent with 01_generate_example_input_regions.py
    #######################################################################################

    # Cell name
    impute_cell_name = 'HFF'

    # epigenomics names
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']

    # The regions you need to impute (must >= 250Kb)
    ch_coord = {'chr2': (23850000, 24100000)}

    # The HiC cell lines to use
    HiC_cell_lines = ['HFF']

    # temp folder for storage
    temp_folder = f'{my_path}/temp'

    #######################################################################################
    #  The below lines are all you need to set
    #######################################################################################

    # models to use
    model = CAESAR()
    model.load_weights(f'HFF_contact_profile.h5')
    model_loop = CAESAR_loop()
    model_loop.load_weights('HFF_loop.h5')

    #######################################################################################
    #  The above lines are all you need to set
    #######################################################################################

    # Step 1: Split the region into small 250 Kb matrices
    generator = training_data_generator(ch_coord, HiC_cell_lines, impute_cell_name, epi_names,
                                        processed_path=processed_path,
                                        pos_enc_dim=8, n_epoches=1, batch_size=1)
    for epoch, batch, (hics, epis, pos_enc, mask), micros, kys in generator:
        ky = kys[0]
        print(ky)

        np.save(f'example_inputs/{ky}_hic.npy', hics)
        np.save(f'example_inputs/{ky}_epi.npy', epis)
        np.save(f'example_inputs/{ky}_micro.npy', micros)
        np.save(f'example_inputs/{ky}_pos_enc.npy', pos_enc)
        np.save(f'example_inputs/{ky}_mask.npy', mask)


