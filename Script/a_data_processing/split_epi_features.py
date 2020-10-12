import pyBigWig
import numpy as np
import os


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


def load_bigwig_for_one_region(path, chromosome, start_pos, end_pos, resolution, output_path=None):
    """
    Load bigwig file and save the signal as a 1-D numpy array

    Args:
        path (str): recommended: {cell_type}_{assay_type}_{reference_genome}.bigwig (e.g., mESC_CTCF_mm10.bigwig)
        start_pos (int):
        end_pos (int):
        resolution (int):
        output_path (str): recommended: {cell_type}_{assay_type}_{reference_genome}_{resolution}bp.npy

    Return:
        vec (numpy.array):
    """
    nBins = (end_pos - start_pos) // resolution
    bw = pyBigWig.open(path)
    vec = bw.stats(chromosome, start_pos, end_pos, exact=True, nBins=nBins)
    for i in range(len(vec)):
        if vec[i] is None:
            vec[i] = 0
    if output_path is not None:
        np.save(output_path, vec)
    return vec


def load_bedGraph_for_one_region(path, chromosome, start_pos, end_pos, resolution, output_path=None, score_column=5):
    # score_column: which column is score (The column indices start from 1.)
    assert start_pos % resolution == 0
    assert end_pos % resolution == 0
    nBins = (end_pos - start_pos) // resolution
    epi_signal = np.zeros((nBins, ))

    f = open(path)
    for line in f:
        lst = line.strip().split()
        ch, p1, p2, v = lst[0], int(lst[1]), int(lst[2]), float(lst[score_column - 1])
        if ch != chromosome or p1 < start_pos or p2 >= end_pos:
            continue
        pp1, pp2 = p1 // resolution, int(np.ceil(p2 / resolution))
        for i in range(pp1, pp2):
            value = (min(p2, (i + 1) * resolution) - max(p1, i * resolution)) / resolution * v
            epi_signal[i - start_pos // resolution] += value
    if output_path is not None:
        np.save(output_path, epi_signal)
    return epi_signal


def process_epi(file, cell_line, epi_name, output_dir):
    resolution = 200
    if type == 'mESC':
        rg = 'mm10'
    else:
        rg = 'hg38'
    chrom_sizes = load_chrom_sizes(rg)
    chroms = load_chrom_sizes(rg)
    del chroms['chrY']

    for ch in chroms.keys():
        if not os.path.exists(f'{output_dir}/{ch}'):
            os.mkdir(f'{output_dir}/{ch}')
        if file.lower().endswith('bigwig'):
            load_bigwig_for_one_region(
                file,
                ch, 0, chrom_sizes[ch],
                resolution=resolution,
                output_path=f'{output_dir}/{cell_line}_{epi_name}_{resolution}bp.npy')
        elif file.lower().endswith('bedgraph'):
            load_bedGraph_for_one_region(
                file,
                ch, 0, chrom_sizes[ch],
                resolution=resolution,
                output_path=f'{output_dir}/{cell_line}_{epi_name}_{resolution}bp.npy')


if __name__ == "__main__":
    file = '../raw_data/epi/IMR90_ATAC_seq_hg38.bedGraph'
    process_epi(file, 'IMR-90', 'ATAC_seq', '../processed_data/Epi/IMR90/')

