import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


processed_hic_path = {
    'hESC': '/data/bulkHiC/H1-hESC/processed',
    'IMR-90': '/data/bulkHiC/IMR90/processed',
    'GM12878': '/data/bulkHiC/GM12878/processed',
    'HFF': '/data/bulkHiC/Dekker_HFF/processed',
    'K562': '/data/bulkHiC/K562/processed'
}
out_path = '/data/microC/high_res_map_project_training_data/HiC/'
chromosomes = [f'chr{i}' for i in list(range(1, 23)) + ['X']]


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


def hic_counts(cell_lines, raw_path, info_dir):
    chroms = [f'chr{i}' for i in list(range(1, 23)) + ['X']]
    depth = pd.DataFrame(np.zeros((len(chroms), len(cell_lines))), index=chroms, columns=cell_lines)
    for ch in chroms:
        fs = [f'{raw_path}/{cell_line}/{ch}_200bp.txt' for cell_line in cell_lines]
        # count
        for cell_line, f in zip(cell_lines, fs):
            print('Counting:', ch, cell_line)
            df = pd.read_csv(f, sep='\s+', names=['pos1', 'pos2', 'count'])
            # print(df)
            # print(df['count'])
            total_count = df['count'].sum()
            depth.loc[ch, cell_line] = total_count
    depth.to_csv(f'{info_dir}/hic_depth.csv', sep='\t')


def HiC_to_strata(cell_lines, info_path, raw_path, processed_path, max_distance=250000, rg='hg38'):
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    # Count!
    hic_counts(cell_lines, raw_path, info_path)

    for cell_line in cell_lines:
        if not os.path.exists(os.path.join(processed_path, cell_line)):
            os.mkdir(os.path.join(processed_path, cell_line))

    chrom_lengths = load_chrom_sizes(rg)
    del chrom_lengths['chrY']
    chroms = chrom_lengths.keys()

    depth = pd.read_csv(f'{info_path}/hic_depth.csv', sep='\t', index_col=0)
    avg_depth = depth.sum(axis=1) / len(cell_lines)  # always use the average

    resolution = 1000
    n_strata = max_distance // resolution

    for ch in chroms:
        avg = avg_depth[ch]
        fs = [f'{raw_path}/{cell_line}/{ch}_200bp.txt' for cell_line in cell_lines]
        # count
        for cell_line, f in zip(cell_lines, fs):
            if not os.path.exists(os.path.join(processed_path, cell_line, ch)):
                os.mkdir(os.path.join(processed_path, cell_line, ch))

            print(ch, cell_line)
            coef = avg / depth.loc[ch, cell_line]
            strata = [np.zeros((int(np.ceil(chrom_lengths[ch] / resolution)) - i,)) for i in range(n_strata)]

            count = 0
            for line in open(f):
                if count % 3000000 == 0:
                    print(' Counting: Line: {0}'.format(count))
                count += 1
                lst = line.strip().split()
                p1, p2, v = lst[0], lst[1], lst[2]
                p1, p2, v = int(p1) // resolution, int(p2) // resolution, float(v)
                if p1 > p2:
                    p1, p2 = p2, p1
                if p2 - p1 < n_strata:
                    strata[p2 - p1][p1] += v * coef

            for i in range(n_strata):
                np.save(os.path.join(processed_path, cell_line, ch, f'{ch}_{resolution}bp_strata_{i}.npy'), strata[i])


if __name__ == '__main__':
    cell_lines = ['hESC', 'HFF']
    info_path = '02_micro_stats'
    raw_path = '/raw/data/path'
    processed_path = '/processed_data/HiC'

    HiC_to_strata(cell_lines, info_path, raw_path, processed_path, max_distance=250000, rg='hg38')
