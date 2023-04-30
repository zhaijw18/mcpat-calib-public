## McPAT-Calib: A RISC-V BOOM Microarchitecture Power Modeling Framework
We make slight improvements to the original McPAT to support the fast analytical power modeling of 7 nm RISC-V BOOM Core.
More importantly, we use machine learning (ML) techniques to calibrate the preliminary modeling results, which can greatly improve the modeling accuracy.
In addition, we propose an active learning (AL) sampling algorithm to reduce the construction cost of the ML-based power calibration model.


### Introduction
The modeling flow of McPAT-Calib can be divided into three parts, the focus is on ML calibration:
1) Microarchitecture Simulation: Use the microarchitecture simulator (gem5) to complete the simulation of the given BOOM microarchitecture configuration and benchmark.
2) McPAT Power Modeling: Convert gem5 simulation results to the XML input file required for McPAT, and then use McPAT to complete preliminary power analysis.
3) ML Calibration: The leakage power and dynamic power are calibrated separately using the ML-based power calibration model, and summed to obtain the total power.

In addition, in the data acquisition stage of constructing the power calibration model, more valuable samples can be selected to label using our proposed AL sampling method (i.e. PowerGS) to reduce the modeling cost.


### Usage
Two modes are provided to allow power estimation directly using the power model pre-trained for the RISC-V BOOM, or retraining a new target model for other target processors, both using the configuration file "config.yml" to provide the required data.

0. First, you need to compile the McPAT for preliminary analytical power modeling:
```sh
$ cd mcpat
$ make
$ cd ..
```

1. If you want to use the power model we have trained for the RISC-V BOOM processor, please set "mode: estimation" and provide the "config.json" and "stats.txt" obtained from Gem5 simulation, as well as the "template.xml" required for gem5-to-mcpat-parser, for example:
```sh
$ python main.py -c config/modeling-example.yml
```

2. If you want to calibrate a new model for another CPU design (or use the proposed power modeling methods), set "mode: train"  and provide the required training data, for example:
```sh
$ python main.py -c config/train-example.yml
```


### Reference
Please refer to the following paper when referring to McPAT-Calib:

```
@ARTICLE{TCAD2023-McPAT-Calib,
  title={McPAT-Calib: A RISC-V BOOM Microarchitecture Power Modeling Framework}, 
  author={Zhai, Jianwang and Bai, Chen and Zhu, Binwu and Cai, Yici and Zhou, Qiang and Yu, Bei},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  year={2023},
  volume={42},
  number={1},
  pages={243-256}
}
```
or 
```
@INPROCEEDINGS{ICCAD2021-McPAT-Calib,
  title={McPAT-Calib: A Microarchitecture Power Modeling Framework for Modern CPUs},
  author={Zhai, Jianwang and Bai, Chen and Zhu, Binwu and Cai, Yici and Zhou, Qiang and Yu, Bei},
  booktitle={2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD)}, 
  year={2021},
  pages={1-9}
}
```



If you have any comments, questions, or suggestions, please write to us:

Jianwang Zhai             
zhaijw18@mails.tsinghua.edu.cn