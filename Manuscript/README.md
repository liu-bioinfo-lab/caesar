# Figures

Most of the code for generating these figures require large-scale output data.
Here we provide the original code we used to generate these data.

## Fig.1
Using ``example_region_from_model.py``, we can generate the predicted
contact map for any given region.

This code requires .h5 model files (can be found in ``\Imputation``),
processed epigenomic features and contact maps (can be generated with code under ``\Imputation`` or under ``\Training\a_data_processing``),
and the calculated norm factors of epigenomic features (can be found in ``\Imputation\utils`` or generated with code under ``\Training\a_data_processing``).


## Fig.2
``/fig2/01_SCC/evaluate_SCC.py`` calculates SCC at 200 bp resolution and
store all predicted contact maps into strata at 1 Kb resolution.

#### The stripe caller is under ``/fig2/02_fine_scale/stripe_caller``,
``Quagga_V0.2.py`` calls stripes from .hic files.

Example usage:
```python
# Path for JuiceTools
juicer = '/nfs/turbo/umms-drjieliu/juicer_tools_1.11.04_jcuda.0.8.jar'
# Chromosomes to calculate
chromosomes = ['chr1']

hic_file = '/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/HFF/raw/HFFc6.hic'
thr = 0.01
stripe_caller_all(
    hic_file=hic_file,
    chromosomes=chromosomes,
    output_file='HFF_MicroC_stripes_chr1.bedpe',
    threshold=thr
)
```

All parameters:
  - hic_file (str): file path
  - reference_genome (str): reference genome
  - chromosomes (list): which chromosomes to calculate
  - output_file (str): output bedpe path
  - threshold (float): p value threshold
  - max_range (int): max distance off the diagonal to be calculated
  - resolution (int): resolution
  - min_length (int): minimum length of stripes
  - min_distance (int): threshold for removing stripes too far away from the diagonal
  - stripe_width (int): stripe width (# of bins at the given resolution)
  - merge (int): merge stripes which are close to each other (# of bins)
  - window_size (int): size of the window for calculating enrichment score

#### The loop caller is under ``/fig2/02_fine_scale/loop_caller``,

``CAESAR_loop_v0.1.py`` calls loops from processed strata.

The input folder stores all strata of one chromosome
in .npy format (numpy.array) as "strata_1kb_{i}.npy".

Example usage:
```python
# Path for JuiceTools
micro_loops = '../../../11_figures/14_call_loops/chr2_micro'
call_loops(folder=micro_loops, output='HFF_micro_chr2.txt')
```

All parameters:
  - max_distance (int): max distance off the diagonal to be calculated
  - resolution (int): resolution
  - smooth_window (int): the window size for smoothing the contact map (# of bins at the given resolution)
  - window_size (int): size of the window for calculating enrichment score (# of bins)
  - peak_size (int): loop size (# of bins)
  - q_threshold (float): p value threshold
  - p_threshold (float): q value threshold
  - merge_distance (int): merge stripes which are close to each other (# of bins)

## Fig.3

The code for training CAESAR with different inputs
(with surrogate Hi-C, with downsampled Hi-C, with 13/7/3 epigenomic features)
and evaluate with SCC.

The example SCC of chromosome 1 is also given.

## Fig.4

Using ``example_region_from_model.py``, we can generate the predicted
contact map for any given region.

'chr8': (127600000, 127850000): MYC

'chrX': (48600000, 48850000): GATA1

## Fig.5
Using ``/examples/example_region_from_model.py``, we can generate the predicted
contact map for all tissues.

## Fig.6
### Attributing results toward epigenomic features
In ``/fig6/``, we provide the code for attribution an arbitrary region toward the input epigenomic features.
With provided coordinates and trained model, the users can calculate the attribution
from each locus of each epigenomic feature.

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

