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


def load_Micro_strata(chrom_file, chrom_length, res=1000, n_strata=200):
    strata = [np.zeros((chrom_length // res + 1 - i,)) for i in range(n_strata)]  # first N strata
    print('Loading strata from {0} at {1} resolution ...'.format(chrom_file, res))
    count = 0
    for line in open(chrom_file):
        if count % 20000000 == 0:
            print(' Line: {0}'.format(count))
        count += 1
        [p1, p2, v] = line.strip().split()
        p1, p2, v = int(p1) // res, int(p2) // res, float(v)
        if p1 > p2:
            p1, p2 = p2, p1
        if p2 - p1 >= n_strata:
            continue
        strata[p2 - p1][p1] += v
    for i in range(n_strata):
        strata[i] = strata[i] / np.mean(strata[i])
        # print(f'Strata {i} - max:{np.max(strata[i])}, std:{np.std(strata[i])}')
    return strata


def load_horizontal_mat(strata, chrom_length, res=1000, n_strata=200):
    mat = np.zeros((chrom_length // res + 1, n_strata))
    for i in range(n_strata):
        mat[:len(strata[i]), i] = strata[i]
    # print(mat.shape)
    return mat


def load_vertical_mat(strata, chrom_length, res=1000, n_strata=200):
    mat = np.zeros((chrom_length // res + 1, n_strata))
    for i in range(n_strata):
        mat[i:i+len(strata[i]), i] = strata[i]
    # print(mat.shape)
    return mat


def pick_max_positions(mat, interval=15, distance_range=(3, 100),
                       line_width=1, window_size=10):
    st, ed = distance_range
    length = mat.shape[0]
    stats = np.sum(mat[:, st:ed], axis=1)
    all_pos = []
    for i in range(0, length, interval):
        region = stats[i: min(i + interval, length)]
        idx = int(np.argmax(region) + i)
        if idx < window_size or idx >= mat.shape[0] - window_size:
            continue
        previous = stats[max(0, idx - interval): idx-1]
        later = stats[idx + 2: min(idx + interval + 1, length)]
        if stats[idx] > np.max(previous) and stats[idx] > np.max(later):
            check = enrichment_score(mat, idx, line_width, (st, ed), window_size)
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
    half = int(line_width // 2)
    x1, x2 = idx - half, idx - half + line_width

    new_mat = np.zeros((distance_range[1] - distance_range[0],))
    for j in range(distance_range[0], distance_range[1]):
        if j < window_size + half or j >= mat.shape[1] - window_size - half:
            continue
        y = j - distance_range[0]
        line_min = np.median(np.concatenate(
            [mat[x1:x2, j-window_size-half:j-half], mat[x1:x2, j+1+half:j+window_size+half+1]]
        ))
        neighbor_mean = max(np.mean(mat[idx-window_size:x1, j-window_size-half:j+window_size+half+1]),
                            np.mean(mat[x2+1:idx+window_size+1, j-window_size-half:j+window_size+half+1]))
        new_mat[y] = line_min / (neighbor_mean + 1e-9) - 1
    # print(np.max(new_mat))
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


def find_stripes(chrom_files, output, reference_genome='hg38',
                 resolution=1000, max_distance=150000,
                 stripe_width=1, window_size=10,
                 shortest=40000, score_threshold=70
                 ):
    chrom_lengths = load_chrom_sizes(reference_genome)
    n_strata = max_distance // resolution

    lst = []
    for ch in chrom_files:
        print(ch)
        print('Loading contact map...')
        strata = load_Micro_strata(chrom_file=chrom_files[ch], chrom_length=chrom_lengths[ch],
                                   res=resolution, n_strata=n_strata + window_size + 1)
        for direction in ['h', 'v']:
            if direction == 'h':
                mat = load_horizontal_mat(strata, chrom_lengths[ch], resolution, n_strata + window_size + 1)
            else:
                mat = load_vertical_mat(strata, chrom_lengths[ch], resolution, n_strata + window_size + 1)

            print('Identifying candidate stripes...')
            max_positions = pick_max_positions(mat, interval=15, distance_range=(n_strata // 30, n_strata),
                                               line_width=stripe_width, window_size=stripe_width)
            print(len(max_positions))

            print('Calculating stripe scores...')
            for pos in max_positions:
                enr = enrichment_score2(mat, pos, line_width=stripe_width,
                                        distance_range=(n_strata // 30, n_strata), window_size=window_size)
                head, tail, score = find_max_slice(enr)
                if tail - head > shortest // resolution and pos > 2 * window_size and score > score_threshold:
                    lst.append((ch, direction, pos * resolution, head, tail, score))

    file = open(output, 'w')
    for (ch, direction, pos, head, tail, score) in lst:
        file.write(f'{ch}\t{direction}\t{pos}\t{head}\t{tail}\t{score}\n')


chrs = ['chr2', 'chr5', 'chr8', 'chr11', 'chr14', 'chr15', 'chr21', 'chr22']
# chrs = ['chr15']
files = {ch: f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/human/raw/HFF/{ch}_200bp.txt' for ch in chrs}
find_stripes(files, 'all_test_set.txt')

