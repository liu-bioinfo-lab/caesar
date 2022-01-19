from os import path
import time
from data_utils import training_data_generator
from CAESAR_model import CAESAR


HUMAN_CHR_SIZES = {'chr1': (100000, 248900000), 'chr2': (100000, 242100000), 'chr3': (100000, 198200000),
                   'chr4': (100000, 190100000), 'chr5': (100000, 181400000), 'chr6': (100000, 170700000),
                   'chr7': (100000, 159300000), 'chr8': (100000, 145000000), 'chr9': (100000, 138200000),
                   'chr10': (100000, 133700000), 'chr11': (100000, 135000000), 'chr12': (100000, 133200000),
                   'chr13': (100000, 114300000), 'chr14': (100000, 106800000), 'chr15': (100000, 101900000),
                   'chr16': (100000, 90200000), 'chr17': (100000, 83200000), 'chr18': (100000, 80200000),
                   'chr19': (100000, 58600000), 'chr20': (100000, 64400000), 'chr21': (100000, 46700000),
                   'chr22': (100000, 50800000), 'chrX': (100000, 156000000)
                   }


def train_and_evaluate(epoches=5, batch_size=10, checkpoint_frequency=1):
    # Build current model
    model = CAESAR()

    j = 0
    my_path = path.abspath(path.dirname(__file__))
    while path.exists('{0}/temp_model_{1}.h5'.format(my_path, j)):
        j += 1

    if j != 0:
        model.load_weights('{0}/temp_model_{1}.h5'.format(my_path, j))

    ch_coord = {'chr1': (100000, 248900000), 'chr4': (100000, 190100000), 'chr7': (100000, 159300000),
                'chr10': (100000, 133700000), 'chr13': (100000, 114300000), 'chr17': (100000, 83200000),
                'chr18': (100000, 80200000)}
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me2',
                 'H3K4me3',  'H3K27ac', 'H3K27me3']
    generator = training_data_generator(ch_coord, ['hESC'], 'hESC', epi_names,
                                        pos_enc_dim=8, n_epoches=epoches, batch_size=batch_size)

    for epoch, batch, (hics, epis, pos_enc), micros in generator:
        if batch == 1:
            if epoch != 1 and (epoch - 1) % checkpoint_frequency == 0:
                model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoch-1))
        # print(epoch, batch, hics.shape, epis.shape, pos_enc.shape, micros.shape)
        t1 = time.time()
        model.train_on_batch([hics, epis, pos_enc], micros)
        t2 = time.time()
        print(' - Training:', t2 - t1, 's')
        mse = model.evaluate([hics, epis, pos_enc], micros, batch_size=batch_size, verbose=0)
        t3 = time.time()
        print(' - Evaluating:', t3 - t2, 's')
        print(' - MSE:', mse)
    model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoches))


if __name__ == '__main__':
    train_and_evaluate(epoches=100, batch_size=20, checkpoint_frequency=20)

