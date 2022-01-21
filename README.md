# EnhanceNet: Plugin Neural Networks for Enhancing Correlated Time Series Forecasting


This is a PyTorch implementation of EnhanceNet in the following paper: \
Razvan-Gabriel Cirstea, Tung Kieu, Chenjuan Guo, Bin Yang, Sinno Jialin Pan, EnhanceNet: Plugin Neural Networks for Enhancing Correlated Time Series Forecasting
, ICDE 2021. This work is based on [DCRNN](https://arxiv.org/abs/1707.01926) and [Graph WaveNet](https://arxiv.org/abs/1906.00121).
Being familiar with those models is strongly recommended.  


## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* torch
* tables
* future

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
The traffic data files for Los Angeles (METR-LA) can be found [here](https://github.com/liyaguang/DCRNN). 

## Run the Model on METR-LA

For RNN variants there are 4 configuration files which can be found under rnn/data/model. Each configuration 
corresponds to rnn/grnn with and without the dynamic weights. To run any of the models follow the command below,
in addition to add the Dynamic Adjacency Matrix Generation Network add the argument --adaptive_supports=1 at the end of the command. 
```bash
python dcrnn_train.py --config_filename=data/model/data/rnn.yaml
```

For TCN variants run the following command:
```bash
# TCN
python train.py 
# GTCN
python train.py --gcn_bool=True
```
In addition for adding the dynamic weights add --temporal_memory=1.

## Citation

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{cirstea2021enhancenet,
  title={EnhanceNet: Plugin Neural Networks for Enhancing Correlated Time Series Forecasting},
  author={Cirstea, Razvan-Gabriel and Kieu, Tung and Guo, Chenjuan and Yang, Bin and Pan, Sinno Jialin},
  booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)},
  pages={1739--1750},
  year={2021},
  organization={IEEE}
}
```
