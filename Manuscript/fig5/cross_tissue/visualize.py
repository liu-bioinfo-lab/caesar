import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


GTEx_tissues = [
        'Adrenal_Gland',
        'Cells_Cultured_fibroblasts',
        'Heart_Left_Ventricle',
        'Cells_EBV-transformed_lymphocytes',
        'Lung',
        'Pancreas',
        'Colon_Sigmoid',
        'Spleen',
        'Stomach',
        'Testis',
        'Nerve_Tibial',
        'Colon_Transverse'
    ]
cell_lines = [
        'adrenal_gland',
        'GM12878',
        'heart_left_ventricle_f51',
        'IMR90',
        'lung',
        'pancreas',
        'sigmoid_colon_m37',
        'spleen_f53',
        'stomach_f51',
        'testis_m37',
        'tibial_nerve_f51',
        'transverse_colon_f51'
    ]


chrs = [f'chr{i}' for i in list(range(1, 23)) + ['X']]
distance = [50, 100, 150]

signif = np.zeros((12, 12))

plt.figure(figsize=(15, 15))
for i, m_tissue in enumerate(cell_lines):
    print(m_tissue)
    for j, e_tissue in enumerate(GTEx_tissues):
        plt.subplot(12, 12, i * 12 + j + 1)

        mat = np.zeros((101, 101))
        for ch in chrs:
            for dist in distance:
                m = np.load(f'../pile_up_v2/{ch}/{dist}/M_{m_tissue}_E_{e_tissue}.npy')
                mat += m
        mat = mat / len(chrs) / len(distance)

        center = np.sum(mat[45:56, 45:56]) / 121
        signif[i, j] = center
        center = str(center)[:4]

        ax = sns.heatmap(mat, vmax=4, vmin=0, cmap='Reds', square=True, cbar=False,
                         xticklabels=False, yticklabels=False)
        rect = Rectangle((40, 40), 21, 21, linewidth=0.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        if i == j:
            ax.text(5, 20, center, {'family': 'Arial', 'size': 18, 'weight': 'bold'})
        else:
            ax.text(5, 20, center, {'family': 'Arial', 'size': 18})


plt.tight_layout()
plt.savefig('test.png')
plt.show()

