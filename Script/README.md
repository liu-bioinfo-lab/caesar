# Script

## Data processing ``/a_data_processing/``
### process_epi_features
Process bigwig or bedgraph files into numpy arrays for each epi-feature.

The raw data should be in:
```
/raw/data/path/hESC/hESC_ATAC_seq_hg38.bedGraph
          ...      /hESC_H3K4me1_hg38.bigWig
          ...        
/raw/data/path/HFF/HFF_DNase_seq_hg38.bedGraph
          ...     /hESC_H3K4me1_hg38.bedGraph
          ...    
```

The processed files are in:
```  
/processed_data/Epi/hESC/chr1/chr1_200bp_ATAC_seq.npy
          ...                /chr1_200bp_H3K4me1.npy
          ...                /...
          ...           /chr2/chr2_200bp_ATAC_seq.npy
          ...           /chr2/...
          ...           ...
/processed_data/Epi/HFF/...
          ...           ...
```

Example code:
```python
signals = ['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3']
cell_lines = ['hESC', 'HFF']
raw_path = '/raw/data/path'
processed_path = '/processed_data/Epi'

# Step 1: bigWig/bedGraph to npy
bigWig_bedGraph_to_vector(cell_lines, signals, raw_path, processed_path)

# Step 2: normalize epi factors
info_dir = '01_epi_stats'
normalize_epi(signals, cell_lines, processed_path, info_dir)
```

### process_MicroC
Process the entire Micro-C contact map into 1250 strata (200 bp resolution).

The raw data should be in:
```
/raw/data/path/hESC/chr1_200bp.txt
          ...      /chr2_200bp.txt
          ...        
/raw/data/path/HFF/chr1_200bp.txt
          ...     /chr2_200bp.txt
          ...    
```
Each ``.txt`` file is in the format of "position1 - position2 - contacts",
which can be generated with ``dump`` command of JuiceTools.
E.g.,
```
10000 10000 3
10000 1100 8
10000 180000 4
...
```

The processed files are in:
```  
/processed_data/MicroC/hESC/chr1/chr1_200bp_strata_1.npy
          ...                   /chr1_200bp_strata_2.npy
          ...                   /...
          ...              /chr2/...
          ...           
/processed_data/MicroC/HFF/...
          ...              ...
```

Example code:
```python
cell_lines = ['hESC', 'HFF']
info_path = '02_micro_stats'
raw_path = '/raw/data/path'
processed_path = '/processed_data/MicroC'

MicroC_to_strata(cell_lines, info_path, raw_path, processed_path, max_distance=250000, threshold=1.01, rg='hg38')
```

### split_HiC
Process the entire Hi-C contact maps into 200 strata (1 kb resolution).

The raw data should be in:
```
/raw/data/path/hESC/chr1_200bp.txt
          ...      /chr2_200bp.txt
          ...        
/raw/data/path/K562/chr1_200bp.txt
          ...      /chr2_200bp.txt
          ...    
```
Each ``.txt`` file is in the format of "position1 - position2 - contacts",
which can be generated with ``dump`` command of JuiceTools.
E.g.,
```
10000 10000 3
10000 1100 8
10000 180000 4
...
```

The processed files are in:
```  
/processed_data/HiC/hESC/chr1/chr1_1000bp_strata_1.npy
          ...                /chr1_1000bp_strata_2.npy
          ...                   /...
          ...           /chr2/...
          ...           
/processed_data/MicroC/K562/...
          ...               ...
```
```python
cell_lines = ['hESC', 'HFF']
info_path = '02_micro_stats'
raw_path = '/raw/data/path'
processed_path = '/processed_data/HiC'

HiC_to_strata(cell_lines, info_path, raw_path, processed_path, max_distance=250000, rg='hg38')
```


## Model training ``/b_model_training/``
In ``/01_loop/`` and ``/02_contact_profile/``, the model can be trained by running ``task.py``:
```python
processed_path = '/processed_data'  # The same path in the previous steps
HiC_cell_lines = ['HFF']  # If using surrogate Hi-C, you can use ['HFF', 'hESC', 'K562', 'GM12878', 'IMR-90']
MicroC_cell_line = 'HFF'

train_and_evaluate(processed_path, HiC_cell_lines, MicroC_cell_line, epoches=100, batch_size=20, checkpoint_frequency=20)
```

Training requires about 200 Gb RAM and takes about 8 hours.


## Attribution ``/c_attribution/``
We can calculate the attribution for arbitrary regions:
```python
attribution(
    cell_line='HFF',
    coordinate='chr1:153500000-153501000,chr1:153540000-153542000',
    epi_names=['ATAC_seq', 'CTCF', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3'],
    epi_path='../processed_data/Epi/',
    hic_path='../raw_data/HiC/HFF/chr2.txt',
    hic_resolution=1000,
    model1_path='contact_profile_model_49.h5',
    model2_path='loop_model_45.h6',
    verbose=1
    )
```
- cell_line: (str) cell line
- coordinate: (str) coordinate of the selected region (must be within 200 kb)
- epi_names: (list of str) epigenomic features to use
- epi_path: (str) the dir for storing all epi-features (See the recommended storing strategy)
- hic_path: (str) the Hi-C `.txt` file for the corresponding chromosome
- hic_resolution: (int) Hi-C's resolution
- model1_path: (str) the part for predicting contact profile
- model2_path: (str) the part for predicting loops

The result will be outputted as a heatmap.

