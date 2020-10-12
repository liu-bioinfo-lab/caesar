"""Attributions main."""
import numpy as np
import threading
import logging
from .utils import parse_coordinate, model_fn, find_1kb_region, load_all_data, int_grad, save_bigBed
# Update databse
from hirespy.serv.update import update


logger = logging.getLogger(__name__)


def attribution(coordinate, signals, cell_type='mESC', verbose=1, info = {}):
    """Calculate the attributions and output a series of bigBed files

    Args:
        coordinate (str): e.g., chr1:153500000-153501000,chr1:153540000-153542000
        model (keras.models.Model): a loaded model
        cell_type (str): cell type
        verbose (int): whether to print

    Return:
        No return. Output to output_bigBed/.
    """

    # database progress
    update(40, "Identify the chosen region...", info["port"])

    if verbose:
        print(' Identify the chosen region...')
    # Step 1: process the coordinate and check whether it's illegal
    try:
        position = parse_coordinate(coordinate)
    except ValueError as value_error:
        raise(value_error)
    except:
        raise(value_error)

    print(position)

    # Step 2: find the corresponding 200-kb region, return the start coordinate
    [ch, start_pos, p11, p12, p21, p22] = find_1kb_region(position)
    print(ch, start_pos, p11, p12, p21, p22)

    # database progress
    update(45, "Loading data for calculation...", info["port"])

    if verbose:
        print(' Loading data for calculation...')
    # Step 3: Load data for the chosen region
    hic, epi = load_all_data(ch, start_pos, signals, info["main"])

    # database progress
    update(50, "Calculating attributions...(This may take a while)", info["port"])

    if verbose:
        print(' Calculating attributions...')
    # Step 4: Calculate attributions
    attributions = int_grad(hic, epi, [p11, p12, p21, p22], steps=100)
    # return a 1000 * 11 numpy array
    # np.save(f'att_chr7_22975000.npy', attributions)

    # database progress
    update(70, "Saving outputs...", info["port"])

    if verbose:
        print(' Saving outputs...')
    # Step 5: Save them into bed file and convert into bigBed file
    save_bigBed(attributions, signals, ch, start_pos, info["output"])
    return position


if __name__ == '__main__':
    signals = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3',
               'H3K9ac', 'H3K27ac', 'H3K27me3', 'H3K36me3']
    # Step 0: Load the model
    print(' Loading the model...')
    # model = model_fn()
    # model = None
    # Can we keep this model loaded in memory? Then users don't need to wait for this...

    # Next steps:
    # attribution('chr7:23105000-23107000,chr7:23182000-23184000')

