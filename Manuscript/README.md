# Utils
## Attributions
Example code for attribution to two regions - chr2:127,725,000-127,875,000 and chr2:10,100,000-10,350,000.
(Paper Fig.6)

## Fine-scale
Example code for calling loops and stripes at 1 kb resolution.
(Paper Fig.2)

## MYC_GATA1
Example code for imputing the contact map of K562 near MYC and GATA1 region.
(Paper Fig.4)

## SCC
The stratum-adjusted correlation coefficient calculation from a trained model.
(Paper Figs.2&3)


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

