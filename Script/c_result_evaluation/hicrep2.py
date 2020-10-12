import numpy as np
from scipy.signal import convolve
# from scipy.interpolate import interp1d


def hicrep(cell_line, preds, reals, h=5, outputs='hicrep_v38_h5_v2.txt'):
    cors = np.zeros((1000,))
    exps = np.loadtxt(f'{cell_line}_exp_all_200bp.txt')
    for i in range(1000):
        preds[i] = preds[i] * exps[i]
        reals[i] = reals[i] * exps[i]

    window_size = 2 * h + 1
    for i in range(1000):
        print('Calculating:', i)
        pred = np.zeros((len(preds[i]) - window_size + 1, ))
        real = np.zeros((len(preds[i]) - window_size + 1, ))

        for j in range(i - window_size + 1, i + window_size):
            kernel_size = window_size - abs(i - j)
            kernel = np.ones((kernel_size, )) / kernel_size

            if j >= 1000:
                continue

            r_pred = convolve(preds[abs(j)], kernel, mode='valid')
            r_real = convolve(reals[abs(j)], kernel, mode='valid')

            delta_length = len(r_pred) - len(pred)
            if delta_length > 0:
                pred += r_pred[delta_length // 2: - delta_length // 2]
                real += r_real[delta_length // 2: - delta_length // 2]
            elif delta_length == 0:
                pred += r_pred
                real += r_real
            else:
                # delta_length = - delta_length
                print('Impossible!')
                raise ValueError('Conv1D')
                # pred[delta_length // 2: - delta_length // 2] += r_pred
                # real[delta_length // 2: - delta_length // 2] += r_real

        cor = np.corrcoef(pred, real)[0, 1]
        cors[i] = cor
    np.savetxt(outputs, cors)
