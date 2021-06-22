# CAESAR
CAESAR (<ins>C</ins>hromosomal structure <ins>A</ins>nd <ins>E</ins>pigenomic<ins>S</ins> <ins>A</ins>nalyze<ins>R</ins>) 
is a deep learning approach to predict nucleosome-resolution 3D chromatin contact maps from
existing epigenomic features and lower-resolution Hi-C contact maps.

Ref: Feng, F., Yao, Y., Wang, X. Q. D., Zhang, X., & Liu, J. (2020). Connecting high-resolution 3D chromatin organization with epigenomics. bioRxiv.

![GitHub Logo](/Image/CAESAR.png)

## Required environment
- Python-3.6.8
- numpy-1.17.4
- scipy-1.4.1
- keras-2.2.4
- tensorflow-1.13.1
- matplotlib-3.0.2
- pyBigWig-0.3.17
- seaborn-0.9.0
- pandas-0.23.4

All these dependencies can be installed with "pip install" or "conda install" command within several minutes.


## Usage
### Processing data
Before training, we need to prepare 1) Hi-C contact maps, 2) Micro-C contact maps, and 3) epigenomic features.

The required input format for Hi-C and Micro-C contact maps are ``.txt`` files generated by ``dump`` function of JuiceTools.
The resolution of the Micro-C contact map should be 200 bp.
Each ``.txt`` file corresponds to one chromosome in the format of "position1 - position2 - contacts". E.g.,
```
10000 10000 3
10000 10200 8
10000 180600 4
...
```
The required input format for epigenomic tracks are ``.bigWig`` or ``.bedGraph`` format.
Each line in the ``.bedGraph`` format should be "chr - start_pos - end_pos - score". E.g.,
```
chr1    10500   10650   0.157
chr1    10650   10850   0.126
chr1    11050   11100   0.347
...
```

Due to high computational burden, it is impossible to feed the entire contact map into the memory,
and therefore we used a 250 kb sliding window with 50 kb step length along the diagonal
(e.g., 0-250,000; 50,000-300,000; 100,000-350,000; ...) to
select the regions and fed them one by one into the model.
In ``/Script/a_data_processing/``, we provide the code for splitting large contact maps into
250 kb regions and load epigenomic tracks into numpy arrays.
All processed files will take about 1 TB of storage for one cell line.

### Model training
CAESAR's inputs include a lower-resolution Hi-C contact map
and a number of histone modification features
(e.g., H3K4me1, H3K4me3, H3K27ac and H3K27me3), chromatin accessibility (e.g., ATAC-seq), and protein binding profiles (e.g., CTCF).
CAESAR captures the Hi-C contact map as a graph `G` with nodes representing genomic regions of 200 bp long,
weighted edges representing chromatin contacts between the regions,
and N epigenomic features modeled as N-dimensional node attributes.
The architecture of CAESAR includes ordinary 1D convolutional layers which extract local epigenomic patterns along the 1D chromatin fiber,
and graph convolutional layers which extract spatial epigenomic patterns over the neighborhood specified by `G`.

In ``/Script/b_model_training/``, we provide the code for training CAESAR model with the processed data.
We recommend using GPU to train the model, and it might take more than 10 hours.

### Attributing results toward epigenomic features
In ``/Script/c_attribution/``, we provide the code for attribution an arbitrary region toward the input epigenomic features.
With provided coordinates and trained model, the users can calculate the attribution
from each locus of each epigenomic feature.


## Applying CAESAR model
In ``/Model/``, we provide some examples of quickly applying the trained model to impute the region
that you are interested in.

For example, after loading the processed Hi-C contact map and epugenomic features of HFF from ``/Model/sample_data/HFF/``,
we can use ``/Model/impute_regions_with_processed_data.py`` to quickly impute the contact map in region chr2:70,100,000-70,350,000.
The result is visualized in ``/Model/outputs/chr2_70100000.png``:
![GitHub Logo](/Model/outputs/chr2_70100000_pred.png)

The provided models are trained with surrogate Hi-C (averaged from hESC, HFF, GM12878, K562, and IMR-90),
Micro-C of hESC and HFF, and epigenomic features of hESC and HFF.
This is the model we used to impute the high-resolution contact maps of other tissues and cell lines.
### i. impute_regions_with_raw_data.py
```python
hic_path = '/nfs/turbo/data/bulkHiC/Dekker_HFF/processed/chr2_1kb.txt'
epi_names = ['DNase_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
epi_path = [
    f'/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/human_tissues/K562/K562_{i}_hg38.bigWig' for i in epi_names
]

predict_region(
        ch='chr2', pos=70100000, hic_resolution=1000,
        hic_file=hic_path,
        bigwig_files=epi_path, bedgraph_files=None
    )
```
- ch: (str) chromosome.
- pos: (int) coordinate. The model will impute a 250 kb region from `pos` to `pos+250000`.
- hic_resolution: (int) the resolution of provided Hi-C. We recommend 1000 or 5000.
- hic_file: (str) the Hi-C file of the corresponding chromosome (generated by ``dump`` function of JuiceTools).
- bigwig_files/bedgraph_files: (list) the bigwig or bedgraph files for epigenomic features.
One of them should be `None`.
The order is ATAC-seq/DNase-seq, CTCF, H3K4me1, H3K4me3, H3K27ac, H3K27me3.


### ii. impute_regions_with_processed_data.py
```python
import numpy as np
hic = np.load('sample_data/HFF/chr2_1000bp_70100000_70350000.npy')
epi_names = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
epi_features = [np.load(f'sample_data/HFF/chr2_200bp_{i}.npy') for i in epi_names]

predict_region(
        ch='chr2', pos=70100000, hic=hic, epi_features=epi_features
    )
```
- ch: (str) chromosome.
- pos: (int) coordinate. The model will impute a 250 kb region from `pos` to `pos+250000`.
- hic: (numpy.array) the input Hi-C contact map generated by ``/Script/a_data_processing/split_HiC.py``.
- epi_features: (list of numpy.array) the input epigenomic features of the corresponding chromosome generated by ``/Script/a_data_processing/split_epi_features.py``.
The order is ATAC-seq/DNase-seq, CTCF, H3K4me1, H3K4me3, H3K27ac, H3K27me3.

Some sample data are in ``/Model/sample_data/``. The example outputs are in ``/Model/outputs/``.



