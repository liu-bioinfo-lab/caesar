import numpy as np
import matplotlib.pyplot as plt


epis = np.load('chr2_32000000_epi.npy')
plt.figure(figsize=(10, 6))
for i, epi in enumerate(epis):
    # print(epi.shape)
    ax1 = plt.subplot(6, 1, i+1)
    ax1.fill_between(np.arange(1250), 0, epi, color='black')
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.spines['left'].set_visible(False)
    if i != len(epis) - 1:
        ax1.set_xticks([])
        ax1.set_xticklabels([])
    # ax1.axis('off')
    # ax1.xaxis.set_visible(True)
    # plt.setp(ax1.spines.values(), visible=False)
    # ax1.yaxis.set_visible(True)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xlim([-0.5, 1250 - 0.5])
ax1.spines['bottom'].set_visible(True)
# x_ticks = ['', '', '', '', '']
# tick_pos = np.linspace(0, 1249, len(x_ticks))  # 这个坐标其实是不对的 差1个bin 但是为了ticks好看只有先这样了
ax1.set_xticks([])
# ax1.set_xticklabels(x_ticks, fontsize=24)

plt.savefig('chr2_32000000_epi.svg')
plt.show()
