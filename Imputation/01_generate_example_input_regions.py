import os
from time import time
import numpy as np
import pandas as pd
import pyBigWig


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
    rg_path = f'{my_path}/utils/{reference_genome}.chrom.sizes'
    f = open(rg_path)
    lengths = {}
    for line in f:
        [ch, l] = line.strip().split()
        lengths[ch] = int(l)
    return lengths


def hictxt2strata(cell_lines, chroms, out_dir, max_distance=250000, rg='hg38'):
    my_path = os.path.abspath(os.path.dirname(__file__))
    for cell_line in cell_lines:
        if not os.path.exists(os.path.join(out_dir, cell_line)):
            os.mkdir(os.path.join(out_dir, cell_line))

    chrom_lengths = load_chrom_sizes(rg)

    resolution = 1000
    n_strata = max_distance // resolution
    depth = pd.read_csv(f'{my_path}/utils/hic_depth.csv', sep='\t', index_col=0)
    avg_depth = depth.sum(axis=1) / 5  # always use the average

    for ch in chroms:
        avg = avg_depth[ch]
        fs = [f'{processed_hic_path[cell_line]}/{ch}_1kb.txt' for cell_line in cell_lines]
        # count
        for cell_line, f in zip(cell_lines, fs):
            if not os.path.exists(os.path.join(out_dir, cell_line, ch)):
                os.mkdir(os.path.join(out_dir, cell_line, ch))

            print(ch, cell_line)
            coef = avg / depth.loc[ch, cell_line]
            strata = [np.zeros((int(np.ceil(chrom_lengths[ch] / resolution)) - i,)) for i in range(n_strata)]

            count = 0
            for line in open(f):
                if count % 10000000 == 0:
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
                np.save(os.path.join(out_dir, cell_line, ch, f'{ch}_{resolution}bp_strata_{i}.npy'), strata[i])


def load_bigWig_for_entire_genome(path, name, chroms, rg='hg38', resolution=200, epi_path=''):
    """
        Load bigwig file and save the signal as a 1-D numpy array

        Args:
            path (str): recommended: {cell_type}_{assay_type}_{reference_genome}.bigwig (e.g., mESC_CTCF_mm10.bigwig)
            name (str): the name for the epigenetic mark
            rg (str): reference genome
            resolution (int): resolution
            epi_path (str): the folders for storing data of all chromosomes

        No return value
        """
    bw = pyBigWig.open(path)
    chromosome_sizes = load_chrom_sizes(rg)

    for ch in chroms:
        if not os.path.exists(f'{epi_path}/{ch}/'):
            os.mkdir(f'{epi_path}/{ch}/')
        end_pos = chromosome_sizes[ch]
        nBins = end_pos // resolution
        end_pos = nBins * resolution  # remove the 'tail'

        vec = bw.stats(ch, 0, end_pos, exact=True, nBins=nBins)
        for i in range(len(vec)):
            if vec[i] is None:
                vec[i] = 0
        np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', vec)


def load_bedGraph_for_entire_genome(path, name, chroms, rg='hg38', resolution=200, epi_path=''):
    print(' Loading:', path)
    chromosome_sizes = load_chrom_sizes(rg)
    epi_signal = {ch: np.zeros((chromosome_sizes[ch] // resolution, )) for ch in chroms}

    f = open(path)
    for line in f:
        # print(line)
        lst = line.strip().split()
        ch, p1, p2, v = lst[0], int(lst[1]), int(lst[2]), float(lst[3])
        if not ch.startswith('chr'):
            ch = 'chr' + ch
        if ch not in chroms:
            continue
        # print(ch, p1, p2, v)
        pp1, pp2 = p1 // resolution, int(np.ceil(p2 / resolution))
        # print(pp1, pp2, len(epi_signal[ch]))
        if max(pp1, pp2) >= len(epi_signal[ch]):
            continue
        for i in range(pp1, pp2):
            value = (min(p2, (i + 1) * resolution) - max(p1, i * resolution)) / resolution * v
            # print(pp1, pp2, i, value)
            epi_signal[ch][i] += value
            # print(i, value)

    for ch in chroms:
        if not os.path.exists(f'{epi_path}/{ch}/'):
            os.mkdir(f'{epi_path}/{ch}/')
        np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', epi_signal[ch])


def epigenomic2array(input_folder, epi_names, chroms, ouput_folder):
    for epi_name in epi_names:
        print(' ', epi_name)
        input_path = f'{input_folder}/{impute_cell_name}_{epi_name}_hg38'
        if os.path.exists(f'{input_path}.bedGraph'):
            load_bedGraph_for_entire_genome(f'{input_path}.bedGraph', epi_name, chroms, epi_path=ouput_folder)
        elif os.path.exists(f'{input_path}.bigWig'):
            load_bigWig_for_entire_genome(f'{input_path}.bigWig', epi_name, chroms, epi_path=ouput_folder)
        else:
            raise ValueError(f'{input_path}.bigWig OR {input_path}.bedGraph DOES NOT EXIST!')


if __name__ == '__main__':
    # Get current path
    my_path = os.path.abspath(os.path.dirname(__file__))

    #######################################################################################
    #  The below lines are all you need to set
    #######################################################################################

    # Cell name
    impute_cell_name = 'HFF'

    # In each folder, there are .txt files dumped from the original .hic file with JuiceTools
    # COMMAND: java -jar juicetools.jar dump observed NONE {hic_file} chr1 chr1 BP 1000 chr1_1kb.txt
    # e.g., chr1_1kb.txt, chr2_1kb.txt
    processed_hic_path = {
        'hESC': '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/H1-hESC/processed',
        'IMR-90': '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/IMR90/processed',
        'GM12878': '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/GM12878/processed',
        'HFF': '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/Dekker_HFF/processed',
        'K562': '/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/K562/processed'
    }

    # epigenomic feature path (bigwig or bedgraph)
    # The format must be {bigwig_path}/{impute_cell_name}_{epi_name}_hg38.{bigWig/BedGraph}
    # e.g., {bigwig_path}/HFF_CTCF_hg38.bigWig
    bigwig_path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed'
    epi_names = ['DNase_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']

    # The regions you need to impute (must >= 250Kb)
    ch_coord = {'chr2': (23850000, 24100000)}

    # The HiC cell lines to use
    HiC_cell_lines = ['HFF']

    # temp folder for storage
    temp_folder = f'{my_path}/temp'

    #######################################################################################
    #  The above lines are all you need to set
    #######################################################################################

    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    # Step 1: Load HiC data and store into strata
    print('Step 1: Load HiC data and store into strata')
    hic_folder = f'{temp_folder}/input_hic'
    if not os.path.exists(hic_folder):
        os.mkdir(hic_folder)
    hictxt2strata(HiC_cell_lines, ch_coord.keys(), hic_folder)

    # Step 2: Load epigenomic data and store into arrays
    print('Step 2: Load epigenomic data and store into arrays')
    epi_folder = f'{temp_folder}/input_epi_{impute_cell_name}'
    if not os.path.exists(epi_folder):
        os.mkdir(epi_folder)
    epigenomic2array(bigwig_path, epi_names, ch_coord.keys(), epi_folder)
