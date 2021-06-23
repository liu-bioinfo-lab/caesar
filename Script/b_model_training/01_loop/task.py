from os import path
import time
from data_utils import training_data_generator
from CAESAR_model import CAESAR_loop


def train_and_evaluate(processed_path, HiC_cell_lines, MicroC_cell_line,
                       epoches=200, batch_size=10, checkpoint_frequency=1):
    my_path = path.abspath(path.dirname(__file__))

    # Build current model
    model = CAESAR_loop()

    j = 0
    while path.exists('{0}/temp_model_{1}.h5'.format(my_path, j)):
        j += 1

    if j != 0:
        model.load_weights('{0}/temp_model_{1}.h5'.format(my_path, j))

    ch_coord = {'chr1': (100000, 248900000), 'chr4': (100000, 190100000), 'chr7': (100000, 159300000),
                'chr10': (100000, 133700000), 'chr13': (100000, 114300000), 'chr17': (100000, 83200000),
                'chr18': (100000, 80200000)}
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    generator = training_data_generator(ch_coord, HiC_cell_lines, MicroC_cell_line, epi_names,
                                        processed_path=processed_path,
                                        pos_enc_dim=8, n_epoches=epoches, batch_size=batch_size)

    for epoch, batch, (hics, epis, pos_enc, mask), micros in generator:
        if batch == 1:
            if epoch != 1 and (epoch - 1) % checkpoint_frequency == 0:
                model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoch - 1))
        # print(epoch, batch, hics.shape, epis.shape, pos_enc.shape, micros.shape)
        t1 = time.time()
        model.train_on_batch([hics, epis, pos_enc, mask], micros)
        t2 = time.time()
        print(' - Training:', t2 - t1, 's')
        mse = model.evaluate([hics, epis, pos_enc, mask], micros, batch_size=batch_size, verbose=0)
        t3 = time.time()
        print(' - Evaluating:', t3 - t2, 's')
        print(' - MSE:', mse)

    model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoches))


if __name__ == '__main__':
    processed_path = '/processed_data'
    HiC_cell_lines = ['HFF']
    MicroC_cell_line = 'HFF'

    train_and_evaluate(processed_path, HiC_cell_lines, MicroC_cell_line,
                       epoches=100, batch_size=20, checkpoint_frequency=20)

