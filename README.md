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
- tensorflow-2.4.1
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
All processed files will take about 50 GB of storage for one cell line.

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
We recommend using GPU to train the model, and it takes about 10 hours.

### Attributing results toward epigenomic features
In ``/Script/c_attribution/``, we provide the code for attribution an arbitrary region toward the input epigenomic features.
With provided coordinates and trained model, the users can calculate the attribution
from each locus of each epigenomic feature.


## Applying CAESAR model
In ``/Model/``, we provide some examples of quickly applying the trained model to impute the region
that you are interested in.

### i. strata_to_example_regions.py
In this step, we extract a 250-kb region from the data processed in previous steps.

Since the processed data is too large to upload to GitHub, we provide an extracted example region
in ``/Model/example_inputs/`` (chr2:23,850,000-24,100,000) for users to quickly test CAESAR.

(Interpolated) HiC in this region:

![GitHub Logo](/Model/example_outputs/chr2_23850000_hic.png)


### ii. example_region_from_model.py
In this step, the extracted data are fed into the model to generate predicted contact maps.
The outputs are visualized in ``/Model/example_outputs/``.


The predicted region chr2:23,850,000-24,100,000:

![GitHub Logo](/Model/example_outputs/chr2_23850000_pred.png)

Comparing with the ground truth - real Micro-C contact map:

![GitHub Logo](/Model/example_outputs/chr2_23850000_micro.png)


