"""Attributions main."""
import numpy as np
from utils import parse_coordinate, find_250kb_region, load_all_data, int_grad, save_bigBed
import matplotlib.pyplot as plt
import seaborn as sns


def attribution(cell_line, coordinate, epi_names,
                epi_path, hic_path, hic_resolution, model1_path, model2_path,
                verbose=1):
    # coordinate (str): e.g., chr1:153500000-153501000,chr1:153540000-153542000
    if verbose:
        print(' Identify the chosen region...')
    # Step 1: process the coordinate and check whether it's illegal
    try:
        position = parse_coordinate(coordinate)
    except:
        raise ValueError
    print(position)

    # Step 2: find the corresponding 200-kb region, return the start coordinate
    [ch, start_pos, p11, p12, p21, p22] = find_250kb_region(position)
    print(ch, start_pos, p11, p12, p21, p22)

    if verbose:
        print(' Loading data for calculation...')
    # Step 3: Load data for the chosen region
    hic, epi = load_all_data(cell_line, ch, start_pos, epi_names, hic_path, hic_resolution, epi_path)

    if verbose:
        print(' Calculating attributions...')
    # Step 4: Calculate attributions
    attributions = int_grad(hic, epi, [p11, p12, p21, p22], steps=100,
                            model1_path=model1_path, model2_path=model2_path)
    # return a 1000 * 11 numpy array
    # np.save(f'att_chr7_22975000.npy', attributions)

    # if verbose:
    #     print(' Saving outputs...')
    # Step 5: Save them into bed file and convert into bigBed file
    # save_bigBed(attributions, signals, ch, start_pos)
    # return position

    if verbose:
        print(' Visualizing outputs...')
    plt.figure()
    max_ = np.quantile(attributions, 0.999)
    g = sns.heatmap(attributions, vmax=max_, vmin=-max_, cmap='coolwarm', yticklabels=epi_names)
    # plt.yticks(fontsize=14)
    g.set_xticks(np.linspace(0, 1250, 6))
    g.set_xticklabels([str(start_pos), '', '', '', '', str(start_pos + 250000)])
    plt.tight_layout()
    plt.savefig(f'Attribution_{ch}_{start_pos}.png')
    plt.close()
    return attributions


if __name__ == '__main__':
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    model1_path = 'contact_profile_model_49.h5'
    model2_path = 'loop_model_45.h6'
    coordinate = 'chr1:153500000-153501000,chr1:153540000-153542000'
    epi_path = '../processed_data/Epi/'
    hic_path = '../raw_data/HiC/HFF/chr2.txt'
    hic_resolution = 1000
    attribution(cell_line='HFF',
                coordinate='chr1:153500000-153501000,chr1:153540000-153542000',
                epi_names=['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3'],
                epi_path='../processed_data/Epi/',
                hic_path='../raw_data/HiC/HFF/chr2.txt',
                hic_resolution=1000,
                model1_path='contact_profile_model_49.h5',
                model2_path='loop_model_45.h6',
                verbose=1)

