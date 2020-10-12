import numpy as np
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
MOUSE_CHR_SIZES = {'chr1': (3000000, 195300000), 'chr2': (3100000, 182000000), 'chr3': (3000000, 159900000),
                   'chr4': (3100000, 156300000), 'chr5': (3000000, 151700000), 'chr6': (3100000, 149500000),
                   'chr7': (3000000, 145300000), 'chr8': (3000000, 129300000), 'chr9': (3000000, 124400000),
                   'chr10': (3100000, 130500000), 'chr11': (3100000, 121900000), 'chr12': (3000000, 120000000),
                   'chr13': (3000000, 120300000), 'chr14': (3000000, 124800000), 'chr15': (3100000, 103900000),
                   'chr16': (3100000, 98100000), 'chr17': (3000000, 94800000), 'chr18': (3000000, 90600000),
                   'chr19': (3100000, 61300000), 'chrX': (3100000, 170800000)
                   }
chromosome_sizes = {'hESC': HUMAN_CHR_SIZES, 'mESC': MOUSE_CHR_SIZES, 'HFF': HUMAN_CHR_SIZES}


def generate_batches(cell_lines, chromosomes, resolutions, epi_names, epigenetic_data, batch_size, inp_window,
                     hic_path, micro_path):
    idx2pos = {}
    pointer = 0

    for cell_line in cell_lines:
        st_ed_pos = chromosome_sizes[cell_line]
        chr_sizes = {ch: st_ed_pos[ch] for ch in chromosomes}

        for ch in chromosomes:
            st, ed = chr_sizes[ch]
            positions = np.arange(st, ed - 249999, 125000)
            indices = np.arange(pointer, pointer + len(positions) * len(resolutions))
            for i, idx in enumerate(indices):
                # a = i % len(cell_lines)
                b = i // len(resolutions)
                c = i % len(resolutions)
                idx2pos[idx] = (ch, positions[b], resolutions[c], cell_line)
            pointer += len(indices)
            print(ch, pointer)

    n_samples = len(idx2pos)
    sample_ids = np.arange(n_samples)
    np.random.shuffle(sample_ids)
    n_batches = n_samples // batch_size
    nBins = 1250
    max_distance = 250000
    print(n_batches)
    # mouse2human = np.load('../mouse2human_v2.npy')

    for j in range(n_batches):
        print(' Batch', j)

        batch_ids = sample_ids[j * batch_size: (j + 1) * batch_size]

        epi = np.zeros((batch_size, nBins + inp_window - 1, len(epi_names)))
        micros = np.zeros((batch_size, nBins, nBins))
        hics = np.zeros((batch_size, nBins, nBins))

        for i, idx in enumerate(batch_ids):
            ch, pos, res, cell_line = idx2pos[idx]
            epi[i, :, :] = epigenetic_data[cell_line][ch][pos // 200 - (inp_window // 2): pos // 200 + 1250 + (inp_window // 2), :]
            micros[i, :, :] = convolve2d(np.load(
                f'{micro_path}/{cell_line}/{ch}/{ch}_200bp_{pos}_{pos + max_distance}.npy'),
                np.ones((3, 3)) / 9, mode='same')
            hics[i, :, :] = np.load(f'{hic_path}/{cell_line}/{ch}/{ch}_{res}bp_{pos}_{pos + max_distance}.npy')

        yield (epi, hics), micros

