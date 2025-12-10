# Official Repository of How (Mis)calibrated is Your Federated CLIP and What To Do About It?

[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2512.04305)



## How to install

### Create your environment:

```bash
$ conda create -n fl2ora python=3.10.8
$ conda activate fl2ora
$ pip install -r requirements.txt
```

### Training and Evaluation
Please run the following commands to `train` and `evaluate' the model:

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
