# Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment

Tianhe Wu*, Shuwei Shi*, Haoming Cai, Mingdeng Cao, Jing Xiao, Yinqiang Zheng and Yujiu Yang

[Tsinghua University Intelligent Interaction Group](https://sites.google.com/view/iigroup-thu/home)

:rocket:  :rocket:  :rocket: **Updates:**
- ✅ **June. 7, 2023**: We release the Assessor360 source code.

[![paper](https://img.shields.io/badge/arXiv-Paper-green.svg)](https://arxiv.org/abs/2305.10983)
[![IIGROUP](https://img.shields.io/badge/IIGROUP-github-red.svg)](https://github.com/IIGROUP)


This repository is the official PyTorch implementation of Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment. :fire::fire::fire:


|Ground Truth|Distortion 1|Distortion 2|Distortion 3|Distortion 4|
|       :---:       |     :---:        |        :-----:         |        :-----:         |        :-----:         | 
| <img width="200" src="images/GT_1.png">|<img width="200" src="images/dis1_1.png">|<img width="200" src="images/dis2_1.png">|<img width="200" src="images/dis3_1.png">|<img width="200" src="images/dis4_1.png">|
|**MOS (GT)**|**3.45 (1)**|**2.95 (2)**|**1.6 (3)**|**1.1 (4)**|
|**Ours (Assessor360)**|**0.5933 (1)**|**0.5213 (2)**|**0.1220 (3)**|**0.0120 (4)**|
| <img width="200" src="images/GT_2.png">|<img width="200" src="images/dis1_2.png">|<img width="200" src="images/dis2_2.png">|<img width="200" src="images/dis3_2.png">|<img width="200" src="images/dis4_2.png">|
|**MOS (GT)**|**4.85 (1)**|**3.25 (2)**|**2.4 (3)**|**1.3 (4)**|
|**Ours (Assessor360)**|**0.9566 (1)**|**0.7263 (2)**|**0.3495 (3)**|**0.0748 (4)**|




## Citation
```
@article{wu2023assessor360,
  title={Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment},
  author={Wu, Tianhe and Shi, Shuwei and Cai, Haoming and Cao, Mingdeng and Xiao, Jing and Zheng, Yinqiang and Yang, Yujiu},
  journal={arXiv preprint arXiv:2305.10983},
  year={2023}
}
```
