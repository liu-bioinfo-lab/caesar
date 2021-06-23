import numpy as np
from scipy.stats import poisson
from scipy.signal import convolve
from statsmodels.sandbox.stats.multicomp import multipletests


def load_horizontal_mat(strata):
    mat = np.zeros((len(strata[0]), len(strata)))
    for i in range(len(strata)):
        s = strata[i]
        mat[:len(s), i] = s
    print('Horizontal:', mat.shape)
    return mat


def load_vertical_mat(strata):
    mat = np.zeros((len(strata[0]), len(strata)))
    for i in range(len(strata)):
        s = strata[i]
        mat[i:i+len(s), i] = s
    print('Vertical:', mat.shape)
    return mat


def call_loops(folder, output):
    max_distance = 200000
    resolution = 1000
    smooth_window = 1
    min_diag_prom = 2
    min_hv_prom = 2
    min_distance = 15
    window_size = 10
    peak_size = 2
    q_threshold = 0.1
    p_threshold = 0.1
    merge_distance = 5
    ch = 'chr2'

    strata = []
    expected_values = []
    for i in range(max_distance // resolution):
        print('Loading strata:', i, end=' ')

        stratum = convolve(
            np.load(f'{folder}/strata_1kb_{i}.npy'),
            np.ones((smooth_window,)) / smooth_window,
            mode='same'
        )
        expected_values.append(np.mean(stratum))
        print(np.mean(stratum), end=' ')
        print(np.max(stratum))
        stratum = stratum / np.mean(stratum)
        strata.append(stratum)
    print(expected_values)
    mat_h = load_horizontal_mat(strata)
    mat_v = load_vertical_mat(strata)

    f = open(output, 'w')
    # for i in range(window_size + peak_size + 1, max_distance // resolution - window_size - peak_size - 1):
    for i in [12, 25, 50, 75, 100, 150]:
        print('Stratum:', i, end=' ')
        stratum = strata[i]

        _filter = np.array([1] * window_size + [0] * (2 * peak_size + 1) + [1] * window_size) / 2 / window_size
        expected = convolve(stratum, _filter, mode='same')

        _filter = np.array([1] * (2 * peak_size - 1)) / (2 * peak_size - 1)
        observed = convolve(stratum, _filter, mode='same')

        # don't be to small!
        expected[expected < 1 / smooth_window / expected_values[i]] = 1 / expected_values[i] / smooth_window

        n_pixels = len(expected)
        p_values = np.ones((n_pixels,))
        for j in range(window_size + peak_size + 1, n_pixels - window_size - peak_size - 1):
            if j % 50000 == 0:
                print(f'  {j} / {n_pixels}')
            if expected[j] == 0:
                continue
            Poiss = poisson(expected[j])
            p_val = 1 - Poiss.cdf(observed[j])
            p_values[j] = p_val

        p_values = np.nan_to_num(p_values, nan=1)  # handle possible NaNs
        q_values = multipletests(p_values, alpha=q_threshold, is_sorted=False, method='fdr_bh')[1]
        q_positions = [i for i in range(len(q_values)) if q_values[i] < q_threshold]

        # merge!
        candidate_positions = []
        temp = None
        for k in range(len(q_positions)):
            if k == 0:
                temp = q_positions[k]
            else:
                if q_positions[k] - temp <= merge_distance:
                    if q_values[temp] > q_values[q_positions[k]]:
                        temp = q_positions[k]
                else:
                    candidate_positions.append(temp)
                    temp = q_positions[k]

        print(f'{len(candidate_positions)} significant contacts (diagonal)')

        for position in candidate_positions:
            xi, yi = position, position + i

            # horizontal
            center = mat_h[xi - peak_size + 1: xi + peak_size, i - peak_size + 1: i + peak_size]
            neighbor1 = mat_h[xi - peak_size + 1: xi + peak_size, i - peak_size - window_size: i - peak_size]
            neighbor2 = mat_h[xi - peak_size + 1: xi + peak_size, i + peak_size + 1: i + peak_size + window_size + 1]
            e1, e2 = max(np.mean(neighbor1), 1 / expected_values[i] / smooth_window), max(np.mean(neighbor2), 1 / expected_values[i] / smooth_window)
            p1 = 1 - poisson(e1).cdf(np.mean(center))
            p2 = 1 - poisson(e2).cdf(np.mean(center))
            p_h = max(p1, p2)

            # vertical
            center = mat_v[yi - peak_size + 1: yi + peak_size, i - peak_size + 1: i + peak_size]
            neighbor1 = mat_h[yi - peak_size + 1: yi + peak_size, i - peak_size - window_size: i - peak_size]
            neighbor2 = mat_h[yi - peak_size + 1: yi + peak_size, i + peak_size + 1: i + peak_size + window_size + 1]
            e1, e2 = max(np.mean(neighbor1), 1 / expected_values[i] / smooth_window), max(np.mean(neighbor2), 1 / expected_values[i] / smooth_window)
            p1 = 1 - poisson(e1).cdf(np.mean(center))
            p2 = 1 - poisson(e2).cdf(np.mean(center))
            p_v = max(p1, p2)

            if not 0 <= p_h < 1:
                continue
            if not 0 <= p_v < 1:
                continue
            if max(p_h, p_v) > p_threshold:
                continue

            # print(xi, yi, p_h, p_v, q_values[position])
            f.write(f'{xi}\t{yi}\t{p_h}\t{p_v}\t{q_values[position]}\n')

        # plt.figure()
        # plt.hist(p_values)
        # plt.show()
        # np.save(f'example_p_{i}.npy', np.array(p_values))
    f.close()
    # loops.sort()
    # f = open(output, 'w')
    # f.write('#chr1\tx1\tx2\tchr2\ty1\ty2\tname\tscore\tstrand1\tstrand2\tcolor\n')
    # for i, (p1, p2, v1, v2, v3) in enumerate(loops):
    #     duplicate = False
    #     if i > 0:
    #         for j in range(i - 1, -1, -1):
    #             pp1, pp2 = loops[j][0], loops[j][1]
    #             if pp1 < p1 - 2:
    #                 break
    #             if abs(p1 - pp1) + abs(p2 - pp2) <= 5:
    #                 duplicate = True
    #                 break
    #     if not duplicate:
    #         # f.write(f'{p1 * 1000}\t{p2 * 1000}\t{v1}\t{v2}\t{v3}\n')
    #         f.write(f'{ch}\t{p1*1000}\t{p1*1000 +1000}\t{ch}\t{p2*1000}\t{p2*1000+1000}\t.\t.\t.\t.\t0,255,255\n')
    # f.close()


if __name__ == '__main__':
    micro_loops = '../../../11_figures/14_call_loops/chr2_micro'
    pred_loops = '../../../11_figures/14_call_loops/chr2_pred'
    call_loops(folder=micro_loops, output='HFF_micro_chr2.txt')
    call_loops(folder=pred_loops, output='HFF_pred_chr2.txt')

    # call_loops(folder='../biorep1_strata', output='biorep1_chr2_noSm.txt')
    # call_loops(folder='../biorep23_strata', output='biorep23_chr2_noSm.txt')


