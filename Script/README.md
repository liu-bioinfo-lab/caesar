# Script

## Data processing ``/a_data_processing/``
### split_MicroC
Split the entire Micro-C contact map into 250 kb squares for training.
```python
cell_type = 'HFF'
input_dir = '../raw_data/HFF/'
output_dir = '../processed_data/MicroC/'
process_MicroC(type=cell_type, input_dir=input_dir, output_dir=output_dir)
```
In `input_dir`, each chromosome has a corresponding `.txt` file named like `chr1.txt`, `chr2.txt`.
Each ``.txt`` file is in the format of "position1 - position2 - contacts". E.g.,
```
10000 10000 3
10000 10200 8
10000 180600 4
...
```
The 250 kb squares will be outputted to `{output_dir}/ch/` as `.npy` format.

### split_HiC
Split the entire Hi-C contact maps into 250 kb squares for training.
```python
input_dirs = [
    '/nfs/turbo/data/bulkHiC/H1-hESC/processed/',
    '/nfs/turbo/data/bulkHiC/K562/processed/',
    '/nfs/turbo/data/bulkHiC/HFF/processed/',
    '/nfs/turbo/data/bulkHiC/GM12878/processed/'
    '/nfs/turbo/data/bulkHiC/IMR90/processed/'
]
output_dir = '../processed_data/HiC/'
process_hics(type='HFF', input_dirs=input_dirs, output_dir=output_dir, resolution=1000)
```
Since we may need to generate surrogate Hi-C, the function allows inputting multiple Hi-C folders.
In each dir of the `input_dirs`, each chromosome has a corresponding `.txt` file named like `chr1.txt`, `chr2.txt`.
Each ``.txt`` file is in the format of "position1 - position2 - contacts". E.g.,
```
10000 10000 3
10000 1100 8
10000 180000 4
...
```
The 250 kb squares will be outputted to `{output_dir}/ch/` as `.npy` format.

### split_epi_features
Process bigwig or bedgraph files into numpy arrays for each epi-feature.
```python
file = '../raw_data/epi/IMR90_ATAC_seq_hg38.bedGraph'
process_epi(file=file, cell_line='IMR-90', epi_name='ATAC_seq', output_dir='../processed_data/Epi/IMR90/')
```
The 200-bp-resolution arrays will be outputted to `{output_dir}/ch/` as `.npy` format.

The recommended storage method:
```
/processed_data/MicroC/cell_line_1/chr1
          ...                     /chr2
          ...                     /chr3
          ...                     ...
/processed_data/MicroC/cell_line_2/chr1
                                  ...
/processed_data/HiC/cell_line_1/chr1
          ...                  ...
/processed_data/HiC/cell_line_2/chr1
          ...                  ...       
/processed_data/Epi/cell_line_1/chr1
          ...                  ...
/processed_data/Epi/cell_line_2/chr1
          ...                  ...
```


## Model training
In loop prediction parts, the loops called at 1 kb resolution should be placed into
`loops_1kb.bedpe` first. Then start training with this command:
```
python task3.py --cell_line HFF --chrs 1,4,7,10,13,17,18 \
    --inp_resolutions 1000 --lr 0.0004 --epoches 50 --batch_size 50 \
    --n_GC_layers 2 --n_GC_units 96 --conv_kernels 96 --conv_windows 3 \
    --checkpoint_frequency 5 \
    --features ATAC_seq,CTCF,H3K4me1,H3K4me3,H3K27ac,H3K27me3 \
    --epi_path /processed_data/Epi/ \
    --hic_path /processed_data/HiC/ \
    --micro_path /processed_data/Micro/
```
- cell_line: comma-separated cell lines (e.g., HFF,hESC)
- chrs: chromosomes for training
- inp_resolutions: Hi-C resolution
- lr: learning rate
- epoches: number of epoches
- batch_size: batch size
- n_GC_layers: number of graph convolutional layers
- n_GC_units: number of graph convolution kernels in each layer
- conv_kernels: comma-separated numbers of 1D convolution kernels in each layer (e.g., 96,96)
- conv_windows: comma-separated sizes of 1D convolution windows in each layer (e.g., 3,3)
- checkpoint_frequency: how often (# of epoches) to save one temp model
- features: epigenomic features to use
- epi_path: path of epigenomic features
- hic_path: path of Hi-C
- micro_path: path of Micro-C

In contact profile predicting part:
```
python task3.py --cell_line HFF --chrs 1,4,7,10,13,17,18 \
    --inp_resolutions 1000 --lr 0.0004 --epoches 50 --batch_size 50 \
    --n_GC_layers 2 --n_GC_units 96 --conv_kernels 96 --conv_windows 15 \
    --checkpoint_frequency 5 \
    --features ATAC_seq,CTCF,H3K4me1,H3K4me3,H3K27ac,H3K27me3 \
    --epi_path /processed_data/Epi/ \
    --hic_path /processed_data/HiC/ \
    --micro_path /processed_data/Micro_residual/
```
The only difference is that the residual contact map (observed Micro-C minus the loop predicting part's output)
should be the new target.


## Attribution


