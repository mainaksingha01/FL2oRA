# Official Repository of How (Mis)calibrated is Your Federated CLIP and What To Do About It?
> [Mainak Singha](https://mainaksingha01.github.io/), [Masih Aminbeidokhti](https://scholar.google.com/citations?hl=en&user=98UoctQAAAAJ), [Paolo Casari](https://scholar.google.com/citations?hl=en&user=CSaXahIAAAAJ), [Elisa Ricci](https://eliricci.eu/), [Subhankar Roy](https://scholar.google.com/citations?user=YfzgrDYAAAAJ&hl=en)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2512.04305)

## Abstract
While vision-language models like CLIP have been extensively studied, their calibration, crucial for reliable predictions, has received limited attention. Although a few prior works have examined CLIP calibration in offline settings, the impact of fine-tuning CLIP in a federated learning (FL) setup remains unexplored. In this work, we investigate how FL affects CLIP calibration and propose strategies to improve reliability in this distributed setting. We first analyze Textual Prompt Tuning approaches and show that they degrade calibration metrics when operating under FL. We also evaluate existing in-training calibration techniques across four global aggregation methods, finding that they provide limited improvements. Our results suggest that the key challenge lies not only in how we aggregate or calibrate, but in which components we choose to fine-tune. Motivated by this insight, we propose $$\text{FL}^2\text{oRA}$$, a straightforward LoRA-based approach that naturally improves calibration in FL, and we analyze the factors behind its effectiveness. Experiments on multiple benchmarks demonstrate that $$\text{FL}^2\text{oRA}$$ consistently produces well-calibrated models, reducing the need for explicit calibration procedures.

## How to install

### Create your environment:

```bash
$ conda create -n fl2ora python=3.10.8
$ conda activate fl2ora
$ pip install -r requirements.txt
```

### Training and Evaluation
Please run the following commands to `train` and `evaluate` the model:

#### To run the in-distribution setting on CIFAR10 and CIFAR100 datasets
```bash
bash flora_cifar.sh cifar10 1 lora 1.0 both
```

#### To run the domain generalization setting on OfficeHome, PACS and VLCS datasets
```bash
bash flora_domain.sh pacs 1 lora 1.0 both
```

#### To run the base-to-new generalization setting on Food101, DTD, Caltech101, Flowers102, and OxfordPets datasets
```bash
bash flora_b2n.sh caltech101 1 lora 1.0 both 16
```

## Citation
If you use our work, please consider citing:
```bibtex
@article{singha2025mis,
  title={How (Mis) calibrated is Your Federated CLIP and What To Do About It?},
  author={Singha, Mainak and Aminbeidokhti, Masih and Casari, Paolo and Ricci, Elisa and Roy, Subhankar},
  journal={arXiv preprint arXiv:2512.04305},
  year={2025}
}
```

## Acknowledgements

Our implementation builds upon the [CoOp](https://github.com/KaiyangZhou/CoOp), [FedOTP](https://github.com/HongxiaLee/FedOTP) and FedPHA(https://github.com/CYFang6/FedPHA) repositories, and we sincerely thank the authors for making their code publicly available.
