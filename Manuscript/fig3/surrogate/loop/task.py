from os import path
import time
from data_utils import training_data_generator
from CAESAR_model import CAESAR_loop


def train_and_evaluate(epoches=200, batch_size=10, checkpoint_frequency=1, loop_epoches=100):
    my_path = path.abspath(path.dirname(__file__))

    # Build current model
    model = CAESAR_loop()
    #
    # j = 0
    # while path.exists('{0}/temp_model_{1}.h5'.format(my_path, j)):
    #     j += 1
    #
    # if j != 0:
    #     model.load_weights('{0}/temp_model_{1}.h5'.format(my_path, j))

    ch_coord = {'chr1': (100000, 248900000), 'chr4': (100000, 190100000), 'chr7': (100000, 159300000),
                'chr10': (100000, 133700000), 'chr13': (100000, 114300000), 'chr17': (100000, 83200000),
                'chr18': (100000, 80200000)}
    epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
    generator = training_data_generator(ch_coord, ['HFF'], 'HFF', epi_names,
                                        pos_enc_dim=8, n_epoches=epoches, batch_size=batch_size, loop_epochs=loop_epoches)

    for epoch, batch, (hics, epis, pos_enc, mask), micros in generator:
        if batch == 0:
            if epoch % checkpoint_frequency == 0 or epoch == loop_epoches:
                model.save_weights('{0}/temp_model_{1}.h5'.format(my_path, epoch))

            if epoch == loop_epoches:  # switch model
                model = CAESAR(nBins=1250, nMarks=6, verbose=1, lr=0.0001, positional_dim=8,
               Epi_kernel=4, Epi_kernel_size=15, Epi_dims=[256, 128], Epi_trainable=True,
               n_GC_layers=2, GC_dim=96, GC_trainable=True,
               n_Conv_layers=1, Conv_dim=96, Conv_size=15, Conv_trainable=True,
               Inner_layer_dims=[512], Inner_trainable=False,
               FC_layer_dims=[], FC_trainable=True)
                model.load_weights('{0}/temp_model_{1}.h5'.format(my_path, epoch))

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
    train_and_evaluate(epoches=100, batch_size=20, checkpoint_frequency=20, loop_epoches=100)

