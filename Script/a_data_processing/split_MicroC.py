import numpy as np
import os
import sys


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


def contact_maps_from_a_processed_file(ch, chrom_file, reference,
                                       oe_norm, chrom_eff_sizes, output_dir):
    """
    Process the contact pair files into numpy arrays
    The files should follow this format:
        position 1 - position 2 - contacts

    Args:
        ch (str): chromosome name
        chrom_file (str): path. E.g. data/chr1_1kb.txt
        reference (str): reference genome
        oe_norm (bool): whether to do OE normalization (each stratum divides by its average value)
        output_dir (str): output path
        chrom_eff_sizes (dict): effective (used) region for all chromosomes {chromosome: (strat_pos, end_pos)}
        resolutions (list): resolutions

    No return value
    """
    lengths = load_chrom_sizes(reference)
    max_distance = 250000
    res = 200

    n_strata = max_distance // res
    # e.g., max = 200 kb, res = 200 bp ==> 1000 bins, 1000 strata
    strata = [np.zeros((lengths[ch] // res + 1 - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata for {0} at {1} resolution ...'.format(ch, res))
    count = 0
    for line in open(chrom_file):
        if count % 500000 == 0:
            print(' Line: {0}'.format(count))
        count += 1
        [p1, p2, v] = line.strip().split()
        p1, p2, v = int(p1) // res, int(p2) // res, float(v)
        if p1 > p2:
            p1, p2 = p2, p1
        if p2 - p1 >= n_strata:
            continue
        strata[p2 - p1][p1] += v
    if oe_norm:
        for i in range(len(strata)):
            strata[i] = strata[i] / np.mean(strata[i])

    if not os.path.exists(f'{output_dir}/{ch}'):
        os.mkdir(f'{output_dir}/{ch}')

    st, ed = chrom_eff_sizes[ch]
    for pos in range(st, ed - max_distance + 1, 125000):
        print('{0} Pos: {1}'.format(ch, pos))
        nBins = max_distance // res
        m = np.zeros((nBins, nBins))
        for i in range(nBins):
            for j in range(i, nBins):
                m[i, j] += strata[j - i][i + pos // res]
                m[j, i] += strata[j - i][i + pos // res]

        fname = f'{output_dir}/{ch}/{ch}_{res}bp_{pos}_{pos+max_distance}.npy'
        np.save(fname, np.log(m + 1))


def process_MicroC(type='mESC', input_dir='../raw_data/', output_dir='../processed_data/'):
    if type == 'mESC':
        sizes = MOUSE_CHR_SIZES
        rg = 'mm10'
    elif type == 'hESC':
        sizes = HUMAN_CHR_SIZES
        rg = 'hg38'
    elif type == 'HFF':
        sizes = HUMAN_CHR_SIZES
        rg = 'hg38'
    else:
        raise ValueError

    chroms = load_chrom_sizes(rg)
    del chroms['chrY']

    for ch in chroms.keys():
        if not os.path.exists(f'{output_dir}/{ch}'):
            os.mkdir(f'{output_dir}/{ch}')
        file_path = os.path.join(input_dir, f'{ch}.txt')
        contact_maps_from_a_processed_file(ch, file_path, rg, True, sizes, output_dir, type)


if __name__ == '__main__':
    cell_type = 'HFF'
    input_dir = '../raw_data/HFF/'
    output_dir = '../processed_data/HFF'
    process_MicroC(cell_type, input_dir, output_dir)
