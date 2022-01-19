import numpy as np


def load_horizontal_mat(folder='../pred_strata', length=182113224):
    mat = np.zeros((length // 1000 + 1, 200))
    for i in range(200):
        print('Loading:', i)
        strata = np.load(f'{folder}/strata_1kb_{i}.npy')
        strata = strata / np.mean(strata)
        mat[:len(strata), i] = strata
    print(mat.shape)
    return mat


def load_vertical_mat(folder='../pred_strata', length=182113224):
    mat = np.zeros((length // 1000 + 1, 200))
    for i in range(200):
        print('Loading:', i)
        strata = np.load(f'{folder}/strata_1kb_{i}.npy')
        strata = strata / np.mean(strata)
        mat[i:i+len(strata), i] = strata
    print(mat.shape)
    return mat


def pick_max_positions(mat, interval=20000, distance_range=(5000, 100000), resolution=1000,
                       line_width=1, window_size=10):
    assert interval % resolution == 0
    assert distance_range[0] % resolution == 0
    assert distance_range[1] % resolution == 0

    st, ed = distance_range[0] // resolution, distance_range[1] // resolution
    size = interval // resolution
    length = mat.shape[0]
    stats = np.sum(mat[:, st:ed], axis=1)
    all_pos = []
    for i in range(0, length, size):
        region = stats[i: min(i + size, length)]
        idx = int(np.argmax(region) + i)
        # print(idx, window_size, mat.shape[0] - window_size)

        if idx < window_size or idx >= mat.shape[0] - window_size:
            continue

        previous = stats[max(0, idx - size): idx-1]
        later = stats[idx + 2: min(idx + size + 1, length)]
        # print(stats[idx], np.max(previous), np.max(later))

        if stats[idx] > np.max(previous) and stats[idx] > np.max(later):
            # print(idx)
            check = enrichment_score(mat, idx, line_width,
                                     (st, ed), window_size)
            if np.sum(check) > 0:
                all_pos.append(idx)
    return all_pos


def enrichment_score(mat, idx, line_width=1, distance_range=(20, 40), window_size=10):
    # st, ed = max(distance_range[0], window_size), min(distance_range[1], mat.shape[1] - window_size)
    half = int(line_width // 2)
    x1, x2 = idx - half, idx - half + line_width

    new_mat = np.zeros((distance_range[1] - distance_range[0],))
    for j in range(distance_range[0], distance_range[1]):
        if j < window_size + half or j >= mat.shape[1] - window_size - half:
            continue
        y = j - distance_range[0]
        line_min = min(np.mean(mat[x1:x2, j-window_size-half:j-half]),
                       np.mean(mat[x1:x2, j+1+half:j+window_size+half+1]))
        neighbor_mean = max(np.mean(mat[idx-window_size:x1, j-window_size-half:j+window_size+half+1]),
                            np.mean(mat[x2+1:idx+window_size+1, j-window_size-half:j+window_size+half+1]))
        new_mat[y] = line_min - neighbor_mean
    return new_mat


def enrichment_score2(mat, idx, line_width=1, distance_range=(5, 100), window_size=10):
    # st, ed = max(distance_range[0], window_size), min(distance_range[1], mat.shape[1] - window_size)
    half = int(line_width // 2)
    x1, x2 = idx - half, idx - half + line_width

    new_mat = np.zeros((distance_range[1] - distance_range[0],))
    for j in range(distance_range[0], distance_range[1]):
        # print(j)
        if j < window_size + half or j >= mat.shape[1] - window_size - half:
            continue
        y = j - distance_range[0]
        # line_min = np.mean(np.concatenate(
        #     [mat[x1:x2, j-window_size-half:j-half], mat[x1:x2, j+1+half:j+window_size+half+1]]
        # ))
        line_min = min(np.mean(mat[x1:x2, j - window_size - half:j - half]),
                       np.mean(mat[x1:x2, j + 1 + half:j + window_size + half + 1]))
        neighbor_mean = max(np.mean(mat[idx-window_size:x1, j-window_size-half:j+window_size+half+1]),
                            np.mean(mat[x2+1:idx+window_size+1, j-window_size-half:j+window_size+half+1]))
        # print(line_min, neighbor_mean)
        new_mat[y] = line_min / (neighbor_mean + 1e-9) - 1
    return new_mat


def find_max_slice(arr):
    _max, head, tail = 0, 0, 0
    _max_ending, h, t = 0, 0, 0
    i = 0
    while i < len(arr):
        _max_ending = _max_ending + arr[i]
        if _max_ending < 0:
            h, t = i + 1, i + 1
            _max_ending = 0
        else:
            t = i + 1
        if _max_ending > _max:
            head, tail, _max = h, t, _max_ending
        i += 1
    return head, tail, _max


def find_stripes():
    print('Loading')
    # mat = load_horizontal_mat(folder='../chr2_hic', length=242193529)
    mat = load_vertical_mat(folder='../chr2_micro', length=242193529)
    print('Filtering')
    max_positions = pick_max_positions(mat)
    print(len(max_positions))

    f = open('hic_chr2_v.txt', 'w')
    for pos in max_positions:
        enr = enrichment_score2(mat, pos, line_width=3, distance_range=(5, 100), window_size=10)
        head, tail, score = find_max_slice(enr)
        if tail - head > 40 and pos > 100 and score > 30:
            f.write(f'{pos * 1000} {head} {tail} {score}\n')
    f.close()


find_stripes()

