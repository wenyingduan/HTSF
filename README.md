# HTSF
Air-quality dataset and code for paper: [Combating Distribution Shift for Accurate Time Series Forecasting via Hypernetworks](https://ieeexplore.ieee.org/document/10077946/) (ICPADS 2022).
## Structure:
- data: including Dongsi, Tiantan, Nongzhanguan and Dingling used in our experiments
- lib: contains self-defined modules such as metrics and data preprocess.
- model: implementation of our model
## Requirements
- python >= 3.8.0
- pytorch >= 1.8
- transformers
- argparse
- configparser
## Run HTSF

```
cd model
python Run.py
```
If you find this code useful in your research, please cite:
(```)
@INPROCEEDINGS{10077946,
  author={Duan, Wenying and He, Xiaoxi and Zhou, Lu and Thiele, Lothar and Rao, Hong},
  booktitle={2022 IEEE 28th International Conference on Parallel and Distributed Systems (ICPADS)}, 
  title={Combating Distribution Shift for Accurate Time Series Forecasting via Hypernetworks}, 
  year={2023},
  volume={},
  number={},
  pages={900-907},
  doi={10.1109/ICPADS56603.2022.00121}}

(```)
