import os
import numpy as np
from scipy.signal import convolve, find_peaks
import matplotlib.pyplot as plt


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


def load_Micro_strata(chrom_file, chrom_length, res=1000, n_strata=200):
    strata = [np.zeros((chrom_length // res + 1 - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata from {0} at {1} resolution ...'.format(chrom_file, res))
    count = 0
    for line in open(chrom_file):
        if count % 20000000 == 0:
            print(' Line: {0}'.format(count))
        count += 1
        [p1, p2, v] = line.strip().split()
        p1, p2, v = int(p1) // res, int(p2) // res, float(v)
        if p1 > p2:
            p1, p2 = p2, p1
        if p2 - p1 >= n_strata:
            continue
        strata[p2 - p1][p1] += v
    for i in range(n_strata):
        strata[i] = strata[i] / np.mean(strata[i])
        # print(f'Strata {i} - max:{np.max(strata[i])}, std:{np.std(strata[i])}')
    return strata


def load_horizontal_mat(strata, chrom_length, res=1000, n_strata=200):
    mat = np.zeros((chrom_length // res + 1, n_strata))
    for i in range(n_strata):
        mat[:len(strata[i]), i] = strata[i]
    # print(mat.shape)
    return mat


def load_vertical_mat(strata, chrom_length, res=1000, n_strata=200):
    mat = np.zeros((chrom_length // res + 1, n_strata))
    for i in range(n_strata):
        mat[i:i+len(strata[i]), i] = strata[i]
    # print(mat.shape)
    return mat


def find_loops(
        chrom_files, output, reference_genome='hg38',
        resolution=1000, max_distance=150000, min_distance=20000,
        loop_intervals=15, prominence=6, threshold=9,
        loop_size=3, window_size=8
):
    chrom_lengths = load_chrom_sizes(reference_genome)
    n_strata = max_distance // resolution
    # t1, t2 = 5, 8  # real HFF
    # t1, t2 = 2, 4  # pred HFF
    # t1, t2 = 7, 12  # loop model
    # t1, t2 = 4, 8  # hic
    # t1, t2 = 5.5, 8.5

    lst = []
    for ch in chrom_files:
        print(ch)
        print('Loading contact map...')
        strata = load_Micro_strata(chrom_file=chrom_files[ch], chrom_length=chrom_lengths[ch],
                                   res=resolution, n_strata=n_strata + window_size + 1)
        mat_h = load_horizontal_mat(strata, chrom_lengths[ch], resolution, n_strata + window_size + 1)
        mat_v = load_vertical_mat(strata, chrom_lengths[ch], resolution, n_strata + window_size + 1)

        print('Identifying peaks...')
        for i in range(min_distance // resolution, n_strata):
            print('Stratum:', i, end=' ')
            stratum = convolve(strata[i], [1 / 3, 1 / 3, 1 / 3], mode='same')
            peaks, _ = find_peaks(stratum, distance=loop_intervals, width=(0, 10), prominence=prominence)
            print(len(peaks))
            for jj, pos in enumerate(peaks):
                h = convolve(mat_h[pos], [1 / 3, 1 / 3, 1 / 3], mode='same')
                peak_h, _ = find_peaks(h, distance=loop_intervals, prominence=prominence)
                if not ((i - 1) in peak_h or i in peak_h or (i + 1) in peak_h):
                    continue

                v = convolve(mat_v[pos + i], [1 / 3, 1 / 3, 1 / 3], mode='same')
                peak_v, _ = find_peaks(v, distance=loop_intervals, prominence=prominence)
                if not ((i - 1) in peak_v or i in peak_v or (i + 1) in peak_v):
                    continue
                val1, val2, val3 = stratum[pos], max(h[i - 1:i + 2]), max(v[i - 1:i + 2])
                if min(val1, val2, val3) < threshold:
                    continue
                lst.append((ch, pos, pos + i, val1, val2, val3))
                # add stats test

    lst.sort()
    f = open(output, 'w')
    for i, (ch, p1, p2, v1, v2, v3) in enumerate(lst):
        duplicate = False
        if i > 0:
            for j in range(i - 1, -1, -1):
                cch, pp1, pp2 = lst[j][0], lst[j][1], lst[j][2]
                if pp1 < p1 - 2 or cch != ch:
                    break
                if abs(p1 - pp1) + abs(p2 - pp2) <= 2:
                    duplicate = True
                    break
        if not duplicate:
            f.write(f'{ch}\t{p1 * 1000}\t{p2 * 1000}\t{v1}\t{v2}\t{v3}\n')
    f.close()


if __name__ == '__main__':
    chrs = ['chr2', 'chr5', 'chr8', 'chr11', 'chr14', 'chr15', 'chr21', 'chr22']
    files = {ch: f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/human/raw/HFF/{ch}_200bp.txt' for ch in chrs}
    find_loops(files, 'HFF_loops_test_set.txt')

