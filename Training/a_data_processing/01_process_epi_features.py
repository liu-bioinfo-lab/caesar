import pyBigWig
import numpy as np
import os
import pandas as pd


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


def load_bigWig_for_entire_genome(path, name, rg, resolution=200, epi_path=''):
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
    if not os.path.exists(f'{epi_path}'):
        os.mkdir(f'{epi_path}')

    chromosome_sizes = load_chrom_sizes(rg)
    del chromosome_sizes['chrY']
    chroms = chromosome_sizes.keys()

    bw = pyBigWig.open(path)

    for ch in chroms:
        end_pos = chromosome_sizes[ch]
        nBins = end_pos // resolution
        end_pos = nBins * resolution  # remove the 'tail'

        vec = bw.stats(ch, 0, end_pos, exact=True, nBins=nBins)
        for i in range(len(vec)):
            if vec[i] is None:
                vec[i] = 0

        if not os.path.exists(f'{epi_path}/{ch}/'):
            os.mkdir(f'{epi_path}/{ch}/')
        np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', vec)


def load_bedGraph_for_entire_genome(path, name, rg, resolution=200, epi_path=''):
    chromosome_sizes = load_chrom_sizes(rg)
    del chromosome_sizes['chrY']
    chroms = chromosome_sizes.keys()
    epi_signal = {ch: np.zeros((chromosome_sizes[ch] // resolution, )) for ch in chroms}

    if not os.path.exists(f'{epi_path}'):
        os.mkdir(f'{epi_path}')

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


def bigWig_bedGraph_to_vector(cell_lines, epi_names, raw_path, processed_path):
    for cell_line in cell_lines:
        for name in epi_names:
            print('HFF', name)
            if name == 'ATAC_seq' or name == 'DNase_seq':
                if os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_ATAC_seq_hg38.bigWig'):
                    file, _type = f'{raw_path}/{cell_line}/{cell_line}_ATAC_seq_hg38.bigWig', 'bw'
                elif os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_DNase_seq_hg38.bigWig'):
                    file, _type = f'{raw_path}/{cell_line}/{cell_line}_DNase_seq_hg38.bigWig', 'bw'
                elif os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_ATAC_seq_hg38.bedGraph'):
                    file, _type = f'{raw_path}/{cell_line}/{cell_line}_ATAC_seq_hg38.bedGraph', 'bg'
                elif os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_DNase_seq_hg38.bedGraph'):
                    file, _type = f'{raw_path}/{cell_line}/{cell_line}_DNase_seq_hg38.bedGraph', 'bg'
                else:
                    raise ValueError(f'{cell_line} - {name}: data not found')
            else:
                if os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_{name}_hg38.bigWig'):
                    file, _type = f'{raw_path}/{cell_line}/{cell_line}_{name}_hg38.bigWig', 'bw'
                elif os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_{name}_hg38.bedGraph'):
                    file, _type = f'{raw_path}/{cell_line}/{cell_line}_{name}_hg38.bedGraph', 'bg'
                else:
                    raise ValueError(f'{cell_line} - {name}: data not found')

            if _type == 'bg':
                load_bedGraph_for_entire_genome(file, name,
                                                rg='hg38', resolution=200, epi_path=f'{processed_path}/{cell_line}')

            else:
                load_bigWig_for_entire_genome(file, name,
                                              rg='hg38', resolution=200, epi_path=f'{processed_path}/{cell_line}')


def get_total_reads(epi_names, cell_lines, processed_path, out_dir):
    chrom_lengths = load_chrom_sizes('hg38')
    del chrom_lengths['chrY']
    resolution = 200

    if not os.path.exists(f'{out_dir}'):
        os.mkdir(f'{out_dir}')

    for k, epi in enumerate(epi_names):
        print(epi)
        depth = pd.DataFrame(np.zeros((len(chrom_lengths), len(cell_lines) * 2)), index=chrom_lengths,
                             columns=cell_lines + [f'density_{elm}' for elm in cell_lines])
        for i, cell_line in enumerate(cell_lines):
            counts = []
            for j, ch in enumerate(chrom_lengths):
                s = np.load(os.path.join(processed_path, cell_line, ch, f'{ch}_{resolution}bp_{epi}.npy'))
                cnt = np.sum(s)
                counts.append(cnt)
                depth.loc[ch, cell_line] = cnt
                depth.loc[ch, f'density_{cell_line}'] = cnt / chrom_lengths[ch]
        depth.to_csv(f'{out_dir}/{epi}_depth.csv', sep='\t')


def normalize_epi(epi_names, cell_lines, processed_path, out_dir):
    get_total_reads(epi_names, cell_lines, processed_path, out_dir)
    norm_factor = pd.DataFrame(np.zeros((len(epi_names), len(cell_lines))), index=epi_names, columns=cell_lines)

    for epi in epi_names:
        print(epi)
        depth = pd.read_csv(f'{out_dir}/{epi}_depth.csv', sep='\t', index_col=0)[cell_lines]
        total_depth = depth.sum(axis=0)
        avg_total_depth = total_depth.mean()
        for cell_line in cell_lines:
            coef = avg_total_depth / total_depth[cell_line]
            norm_factor.loc[epi, cell_line] = coef
    norm_factor.to_csv(f'{out_dir}/epi_norm_factors.csv', sep='\t')


if __name__ == '__main__':
    # signals = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3',
    #            'H3K4me2', 'H3K9me3', 'H3K79me2', 'Nanog', 'Rad21']
    signals = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    cell_lines = ['hESC', 'HFF']

    # Step 1: bigwig2npy
    raw_path = '/raw/data/path'
    processed_path = '/processed_data/Epi'  # Modify this!
    bigWig_bedGraph_to_vector(cell_lines, signals, raw_path, processed_path)

    # Step 2: normalize epi factors
    info_dir = '01_epi_stats'
    normalize_epi(signals, cell_lines, processed_path, info_dir)

