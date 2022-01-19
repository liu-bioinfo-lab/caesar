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


## Using CAESAR model
In ``/Imputation/``, we provide some examples of quickly applying the trained model to impute the region
that you are interested in.

### i. 01_generate_example_input_regions.py
Before this step, users need to get Hi-C files and epigenomic features ready.
The example data can be downloaded at DropBox (link: ).

This .py file generate numpy matrices from Hi-C contact maps and epigenomic features for the next steps.

The parameters users need to specify (from Line 152 of this file):
- impute_cell_name: the cell to impute (must match the file name of bigWig/bedGraph files)
- processed_hic_path: the folders containing input Hi-C in .txt format (generated with JuiceTools dump). File names must be chr{?}_1kb.txt.
- bigwig_path: the folder containing all bigWig/bedGraph files. File names must be {cell_name}_{epi_name}_hg38.bigWig/bedGraph.
- epi_names: ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3'] 
- ch_coord: The regions you need to impute (must >= 250Kb)
- HiC_cell_lines: The Hi-C contact maps to use
- temp_folder: temp folder for storage


```
>>> python 01_generate_example_input_regions.py
...
Step 1: Load HiC data and store into strata
chr2 HFF
 Counting: Line: 0
 Counting: Line: 10000000
 Counting: Line: 20000000
 Counting: Line: 30000000
 Counting: Line: 40000000
 Counting: Line: 50000000
 Counting: Line: 60000000
 Counting: Line: 70000000
 Counting: Line: 80000000
 Counting: Line: 90000000
 Counting: Line: 100000000
Step 2: Load epigenomic data and store into arrays
  DNase_seq
 Loading: /nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed/HFF_DNase_seq_hg38.bedGraph
  CTCF
 Loading: /nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed/HFF_CTCF_hg38.bedGraph
  H3K4me1
 Loading: /nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed/HFF_H3K4me1_hg38.bedGraph
  H3K4me3
 Loading: /nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed/HFF_H3K4me3_hg38.bedGraph
  H3K27ac
 Loading: /nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed/HFF_H3K27ac_hg38.bedGraph
  H3K27me3
 Loading: /nfs/turbo/umms-drjieliu/proj/4dn/data/cut_and_run/HFF/processed/HFF_H3K27me3_hg38.bedGraph
```

### ii. 02_example_region_from_model.py
This .py file generate 250 Kb regions (Hi-C and epi) from the temp outputs of the last step 
and the use CAESAR model to impute 200-bp-resolution Micro-C contact maps.
If the given region is longer than 250 Kb, it will impute multiple contact maps
with a 50 Kb step length.

The parameters users need to specify (from Line 211 of this file):
- surrogate: whether the input Hi-C is a surrogate Hi-C.
If not, it will use the model trained with a matched Hi-C (HFF in this example)

The other parameters have to be consistent with 01_generate_example_input_regions.py.

```
>>> python 02_example_region_from_model.py
...
Finish generating positional encoding... 0.0min:0.0037298202514648438sec
 Loading epigenomics... chr2 DNase_seq 1210967
 Loading epigenomics... chr2 CTCF 1210967
 Loading epigenomics... chr2 H3K4me1 1210967
 Loading epigenomics... chr2 H3K4me3 1210967
 Loading epigenomics... chr2 H3K27ac 1210967
 Loading epigenomics... chr2 H3K27me3 1210967
Finish generating epigenomic features... 0.0min:0.38011598587036133sec
 Loading HiC strata: chr2 23850000 24100000
  0 / 250
  25 / 250
  50 / 250
  75 / 250
  100 / 250
  125 / 250
  150 / 250
  175 / 250
  200 / 250
  225 / 250
Finish loading HiC strata... 0.0min:0.41601085662841797sec
 Converting to matrices: chr2
Finish processing 1 HiC maps... 0.0min:0.12031936645507812sec
Start running:
Epoch: 1  Batch: 1
chr2_23850000
```


### iii. 03_visualize_results.py
This .py file visualize all predicted contact maps.

The parameters users need to specify (from Line 137 of this file):
All parameters have to be consistent with 01_generate_example_input_regions.py.

```
>>> python 03_visualize_results.py
...
chr2 23850000
```

The results are stored in /temp/example_outputs/ as .png files:

(Interpolated) Input HiC in this region:

![GitHub Logo](/Imputation/temp/example_outputs/chr2_23850000_input.png)

The predicted region chr2:23,850,000-24,100,000:

![GitHub Logo](/Imputation/temp/example_outputs/chr2_23850000_pred.png)

Comparing with the ground truth - real Micro-C contact map:

![GitHub Logo](/Imputation/temp/example_outputs/chr2_23850000_micro.png)


### ii. example_region_from_model.py
In this step, the extracted data are fed into the model to generate predicted contact maps.
The outputs are visualized in ``/Model/example_outputs/``.


## Train CAESAR from scratch
In ``/Training/``, we provide the detailed code of training CAESAR
from scratch.

The detailed step-to-step code is in ``/Training/README.md``.

### Processing data
Before training, we need to prepare files for
1) Hi-C contact maps,
2) Micro-C contact maps
3) epigenomic features

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

In ``/Training/a_data_processing/``, we provide the code for loading large contact maps into
strata and load epigenomic tracks into numpy arrays.
All processed files will take about 5 GB of storage for one cell line.

### Model training
CAESAR's inputs include a lower-resolution Hi-C contact map
and a number of histone modification features
(e.g., H3K4me1, H3K4me3, H3K27ac and H3K27me3), chromatin accessibility (e.g., ATAC-seq), and protein binding profiles (e.g., CTCF).
CAESAR captures the Hi-C contact map as a graph `G` with nodes representing genomic regions of 200 bp long,
weighted edges representing chromatin contacts between the regions,
and N epigenomic features modeled as N-dimensional node attributes.
The architecture of CAESAR includes ordinary 1D convolutional layers which extract local epigenomic patterns along the 1D chromatin fiber,
and graph convolutional layers which extract spatial epigenomic patterns over the neighborhood specified by `G`.

Due to high computational burden, it is impossible to feed the entire contact map into the memory,
and therefore we used a 250 kb sliding window with 50 kb step length along the diagonal
(e.g., 0-250,000; 50,000-300,000; 100,000-350,000; ...) to
select the regions and fed them one by one into the model.

In ``/Training/b_model_training/``, we provide the code for splitting large contact maps into
250 kb regions and training CAESAR model with the processed data.
We recommend using GPU to train the model, and it takes about 10 hours.




