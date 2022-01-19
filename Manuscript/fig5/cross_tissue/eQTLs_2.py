import re
import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp2d
from scipy.signal import convolve, convolve2d
from CAESAR_model import CAESAR, CAESAR_loop


HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)}


GTEx_tissues = [
        'Adrenal_Gland',
        'Cells_Cultured_fibroblasts',
        'Cells_EBV-transformed_lymphocytes',
        'Colon_Sigmoid',
        'Colon_Transverse',
        'Heart_Left_Ventricle',
        'Lung',
        'Nerve_Tibial',
        'Ovary',
        'Pancreas',
        'Spleen',
        'Stomach',
        'Testis',
        'Uterus',
        'Vagina'
    ]
cell_lines = [
        'adrenal_gland',
        'GM12878',
        'IMR90',
        'lung',
        'sigmoid_colon_m37',
        'sigmoid_colon_m54',
        'transverse_colon_f51',
        'transverse_colon_f53',
        'transverse_colon_m37',
        'heart_left_ventricle_f51',
        'heart_left_ventricle_f53',
        'tibial_nerve_f51',
        'tibial_nerve_f53',
        'ovary',
        'pancreas',
        'spleen_f51',
        'spleen_f53',
        'stomach_f51',
        'stomach_f53',
        'testis_m37',
        'uterus_f53',
        'vagina_f51'
    ]  # 22 in total


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
    rg_path = re.sub('CAESAR_review.*$', f'CAESAR_review/00_utils/epi_norm_factors_all.csv', my_path)
    df = pd.read_csv(rg_path, sep='\t', index_col=0)

    epigenetic_data = {}
    res = 200
    for cell_line in cell_lines:
        epi_data = {}
        for ch in chromosomes:
            epi_data[ch] = None
            for i, k in enumerate(epi_names):
                norm_factor = df.loc[k, cell_line]
                path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/high_res_map_project_training_data/Epi/cell_lines'
                src = '{0}/{1}/{2}/{3}_{4}bp_{5}.npy'.format(path, cell_line, ch, ch, res, k)
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


def generate_mask(size=1250, distance=10, batch_size=20):
    msk = np.ones((size, size))
    for i in range(size):
        for j in range(max(0, i - distance), min(size, i + distance + 1)):
            msk[i, j] = 0
    batch_msk = np.array([msk for _ in range(batch_size)])
    return batch_msk


def training_data_generator(ch_coord, hic_mats, micro_cell_line, epi_names, pos_enc_dim=8,
                            n_epoches=100, batch_size=20):
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
    epi_features = load_epigenetic_data([micro_cell_line], ch_coord.keys(), epi_names)
    (t_min, t_sec) = divmod(time() - _tm, 60)
    _tm = time()
    print('Finish generating epigenomic features...', '{}min:{}sec'.format(t_min, t_sec))

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

            print(' Batch:', _batch + 1, micro_cell_line)
            batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]
            # print(batch_idx)
            kys = [all_keys[__] for __ in batch_idx]
            print(kys)

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


def load_hic_strata(ch_coord, hic_cell_lines):
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
    return hic_mats


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


def load_micro_strata_denoise(ch_coord, micro_cell_line):
    _tm = time()
    hic_strata = {ch: [] for ch in ch_coord}
    for ch in ch_coord:
        st, ed = ch_coord[ch]
        print(' Loading Micro-C strata:', ch, st, ed)
        for i in range(1250):
            if i % 125 == 0:
                print(' ', i, '/', 1250)
            path = '/nfs/turbo/umms-drjieliu/usr/temp_Fan/filtered_MicroC'
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


def vis_region(x, outpath):
    x = convolve2d(x, np.ones((3, 3)) / 9, mode='same')
    plt.figure(figsize=(9, 9))
    sns.heatmap(x, vmax=np.max(x), vmin=0, cmap='Reds', cbar=True, xticklabels=False, yticklabels=False, square=True)
    plt.savefig(outpath)
    # plt.show()
    plt.close()


def find_pile_up_region(strata, ch, pos1, pos2, padding=10):
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    st, ed = HUMAN_CHR_SIZES[ch]
    pos1, pos2 = (pos1 - st) // 200, (pos2 - st) // 200
    strata_id = pos2 - pos1
    region = np.zeros((padding * 2 + 1, padding * 2 + 1))
    for i in range(- 2 * padding, 2 * padding + 1):
        idx = strata_id + i
        if idx >= 0 and i <= 0:
            st, ed = pos1 - padding - i, pos1 + padding + 1
        elif idx > 0 and i > 0:
            st, ed = pos1 - padding, pos1 + padding - i + 1
        elif idx < 0 and i < 0:
            st, ed = pos1 - padding - i + idx, pos1 + padding + 1 + idx
        else:
            raise ValueError('Impossible!')

        s = strata[abs(idx)][st: ed]
        if i <= 0:
            for x1, x2 in zip(range(-i, 2 * padding + 1), range(len(s))):
                region[x1, x2] = s[x2]
        else:
            for x1, x2 in zip(range(len(s)), range(i, 2 * padding + 1)):
                region[x1, x2] = s[x1]
    return region


def load_positions(file, chromosome='chr7', distance_range=(2000, 100000)):
    f = open(file)
    variants, tss = [], []
    for line in f:
        lst = line.strip().split()
        # print(lst)
        if lst[0] != chromosome:
            continue
        var, ts = int(lst[1]), int(lst[2])
        if distance_range[0] <= abs(var - ts) < distance_range[1]:
            variants.append(var)
            tss.append(ts)
    return variants, tss


def pile_up_analysis(strata, tissue, all_tissues, ch='chr2', padding=50, output_path='pile_up_v1'):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    st, ed = HUMAN_CHR_SIZES[ch]
    for tt in all_tissues:
        names = [50, 100, 150, 200]
        rgs = [(2000, 50000), (50000, 100000), (100000, 150000), (150000, 179000)]
        for rg, nm in zip(rgs, names):
            if not os.path.exists(f'{output_path}/{nm}'):
                os.mkdir(f'{output_path}/{nm}')
            print(rg)
            file = f'/home/fanfeng/CAESAR_review/11_figures/07_eQTL_enrichment/processed/{ch}/{tt}_specific_eQTLs.txt'
            variants, tss = load_positions(file=file, chromosome=ch, distance_range=rg)
            i, length = 0, len(tss)
            count = 0
            piled_up = np.zeros((padding * 2 + 1, padding * 2 + 1))
            for v, t in zip(variants, tss):
                if i % 1000 == 0:
                    print(f' Map:{tissue}, eQTL:{tt} - {i} / {length}')
                i += 1
                if min(v, t) <= st + 250000 or max(v, t) >= ed - 250000:
                    continue
                region = find_pile_up_region(strata, ch, v, t, padding)
                piled_up += region
                count += 1
            piled_up = piled_up / count
            np.save(f'{output_path}/{nm}/M_{tissue}_E_{tt}.npy', piled_up)
            vis_region(piled_up, f'{output_path}/{nm}/fig_M_{tissue}_E_{tt}.png')


if __name__ == '__main__':
    my_path = os.path.abspath(os.path.dirname(__file__))
    model = CAESAR()
    model.load_weights(f'/home/fanfeng/CAESAR_review/11_figures/surrogateHFF_stripe_temp_model_100.h5')

    model_loop = CAESAR_loop()
    model_loop.load_weights(f'/home/fanfeng/CAESAR_review/11_figures/HFF_loop_temp_model_100.h5')

    hic_cell_lines = ['HFF', 'hESC', 'K562', 'GM12878', 'IMR-90']
    ch_coord = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000)}
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    correction = correction_mat()

    outdir = f'{my_path}/pile_up_v2'

    # hESC
    # micro_all_strata = load_micro_strata(ch_coord, 'hESC')
    # for ch in ch_coord.keys():
    #     pile_up_analysis(strata=micro_all_strata[ch], tissue='hESC', all_tissues=GTEx_tissues,
    #                      ch=ch, padding=50, output_path=f'{outdir}/{ch}')

    # HFF
    micro_all_strata = load_micro_strata(ch_coord, 'HFF')
    for ch in ch_coord.keys():
        pile_up_analysis(strata=micro_all_strata[ch], tissue='HFF', all_tissues=GTEx_tissues,
                         ch=ch, padding=50, output_path=f'{outdir}/{ch}')

    micro_strata = load_micro_strata_denoise(ch_coord, 'HFF')
    averages = None
    for ch in ch_coord:
        avg = np.array([np.mean(elm) for elm in micro_strata[ch]])
        if averages is None:
            averages = avg
        else:
            averages = averages + avg
    averages = averages / len(ch_coord)
    np.savetxt(f'{my_path}/HFF_avgs_v2.txt', averages)

    for ch in ch_coord.keys():
        pred_strata = []
        for i in range(1000):
            if i % 125 == 0:
                print(' ', i, '/', 1000)
            s1 = np.load(f'/nfs/turbo/umms-drjieliu/usr/temp_Fan/05_HFF_stripe_denoise_all/{ch}/strata_{i}.npy')
            s2 = np.load(f'/nfs/turbo/umms-drjieliu/usr/temp_Fan/07_HFF_loop_all/{ch}/strata_{i}.npy')
            s = s1 + s2
            s = s / np.mean(s) * averages[i]
            pred_strata.append(s)
        pile_up_analysis(strata=pred_strata, tissue='HFFpred', all_tissues=GTEx_tissues,
                         ch=ch, padding=50, output_path=f'{outdir}/{ch}')

    hic_all_mats = load_hic_strata(ch_coord, hic_cell_lines)

    # all cell lines
    for cell_line in cell_lines:
        generator = training_data_generator(ch_coord, hic_all_mats, cell_line, epi_names,
                                            pos_enc_dim=8, n_epoches=1, batch_size=1)
        pred_all_strata = {ch: [np.zeros(elm.shape) for elm in micro_all_strata[ch]] for ch in micro_all_strata}

        for epoch, batch, (hics, epis, pos_enc, mask), kys in generator:
            _res1 = model.predict([hics, epis, pos_enc])[0, :, :]
            _res2 = model_loop.predict([hics, epis, pos_enc, mask])[0, :, :]
            _res = _res1 + _res2
            ky = kys[0]
            pred_all_strata = update_strata(ch_coord, pred_all_strata, ky, _res, correction)

        for ch in ch_coord.keys():
            for i in range(len(pred_all_strata[ch])):
                pred_all_strata[ch][i] = np.abs(pred_all_strata[ch][i])
                pred_all_strata[ch][i] = pred_all_strata[ch][i] / (np.mean(pred_all_strata[ch][i]) + 1e-15) * averages[i]

            pile_up_analysis(strata=pred_all_strata[ch], tissue=cell_line, all_tissues=GTEx_tissues,
                             ch=ch, padding=50, output_path=f'{outdir}/{ch}')


