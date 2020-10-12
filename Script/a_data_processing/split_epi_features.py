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


if __name__ == "__main__":
    reference_genome = 'hg38'
    chrom_sizes = load_chrom_sizes(reference_genome)
    ch = 'chr1'

    load_bedGraph_for_one_region(
        '../raw_data/epi/IMR90_ATAC_seq_hg38.bedGraph',
        ch, 0, chrom_sizes[ch],
        resolution=200,
        output_path='../processed_data/epi/IMR90_chr1_ATAC_seq_200bp.npy')

    load_bigwig_for_one_region(
        '../raw_data/epi/IMR90_CTCF_hg38.bigWig',
        ch, 0, chrom_sizes[ch],
        resolution=200,
        output_path='../processed_data/epi/IMR90_CTCF_200bp.npy')

    # epigenetic_features = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
    #                        'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    # path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/IMR90'
    # for epi in epigenetic_features:
    #     print(epi)
    #     if epi == 'ATAC_seq':
    #         load_bedGraph_for_one_region('{0}/IMR90_{1}_hg38.bedGraph'.format(path, epi), 'chr1', 0, 248956000, resolution=200,
    #                                      output_path='epi/IMR90_chr1_{0}_200bp.npy'.format(epi))
    #     else:
    #         load_bigwig_for_one_region('{0}/IMR90_{1}_hg38.bigWig'.format(path, epi), 'chr1', 0, 248956000, resolution=200,
    #                                    output_path='epi/IMR90_chr1_{0}_200bp.npy'.format(epi))

