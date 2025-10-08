This package includes Python code to implement the temporal convolutional network (TCN) used in "Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation" by Dean D. Molinaro, Keaton L. Scherpereel, Ethan B. Schonhaut, Georgios Evangelopoulos, Max K. Shepherd, and Aaron J. Young.
Prepared by Dean D. Molinaro (contact: molinarodean@gmail.com)
Last Modified: 3/5/2024

///// Package Contents /////
config_utils.py: a script used for loading config files

dataloader.py: an implementation of a dataset class that dynamically loads trials from our available dataset

example.py: an example script, which loads trials available from our dataset and runs them through the pretrained TCN. This file supports a few additional command line arguments:
--config_path: specify a different config for loading data/pretrained models.
--device: specify a different device aside from the CPU (default) for deploying the model.

tcn.py: the implementation of the TCN class

./configs/default_config.py: the default config for deploying the TCN trained with all sensor inputs

./configs/sensor_selection: a collection of configs for deploying the TCN with varying sets of input sensors

./data: a sample of data from our available dataset

./models: a set of pretrained TCNs used in this study with varying model inputs

///// Required Python Modules /////
This package was prepared using Python 3.7.1. The Python modules and corresponding versions used during the preparation of this package are provided below.

pandas: 1.3.5
torch: 1.6.0
