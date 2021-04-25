# Efficient KDE  via LSH implementation & benchmark

 - Authors: Haoran Pu, Erica Wei

This is the repository for the code implemented and benchmarked for KDE estimation via LSH. For benchmark and implementation.

## Repo Decomposition



 - Original Implementation of Laplacian KDE with Binning Rahimi and Recht LSH from the paper [Space and Time Efficient Kernel Density Estimation in High Dimensions](https://github.com/talwagner/efficient_kde) (`laplacian_original.py`)

 - A faster implementation of Laplacian KDE with Binning RR via GPU (tensorflow) (`laplacian_tensorflow.py`)

 - A Numpy implementation of Gaussian KDE with Near Optimal LSH. (`near_optimal.py`)
 
 - A Tensorflow implementation of Gaussian KDE with Near Optimal LSH (`near_optimal_tf.py`) but find out to be less efficient because tensorflow process tracking ball on grid in an inefficient manner.

- Jupyter notebooks for benchmarking different KDE+LSH pair based on relative error to true KDE as well as query/preprocessing time (`laplacian benchmark.ipynb` and `near optimal benchmark.ipynb`)

- Data used for final report. (`data`, `normal`/`uniform` menas the dataset for benchmarking)

- Inside Data directory, we also include the code for plotting the figure as well as the figure themselves (`data/plot.py`, `data/plot1`, `data/plot2`) 
