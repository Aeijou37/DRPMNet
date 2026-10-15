# DRPMNet

**Dynamic Routing and Progressive Multiscale Feature Fusion for Real-Time Instance Segmentation**

Guojie Li, Wenxia Bao, Wentao Guo, Jianwei Huang, and Zhiyou Cheng

Anhui University

Research code accompanying the DRPMNet manuscript. The method builds on YOLO11s-seg and combines dynamic routing, progressive feature fusion, and a training-only boundary contrastive objective.

> **Release scope:** This repository currently contains a partial source snapshot: three standalone modules, dataset path templates, and a module smoke check. The full training/evaluation pipeline, final model integration, trained weights, annotation converters, and deployment scripts are not included. The manuscript results below have not been reproduced with this partial release. See [release status](paper_docs/release_status.md) for the differences between the supplied snapshot and the final method description.

## Method

- **DRDB — Dynamic Routing Dual-Branch:** combines spatial and dilated-context features with input-conditioned branch weights at the low-resolution feature level. Both branches execute during inference.
- **PMFF — Progressive Multiscale Feature Fusion:** propagates deep semantic information through two successive fusion stages between adjacent scales.
- **EACL — Edge-Aware Contrastive Learning:** uses boundary locations and asymmetric strong/weak feature views during training. Its auxiliary branches are removed for deployment.

The released classes retain their original source names, `DRDBN`, `PMAF`, and `EACL`. In particular, `PMAF` is a legacy fusion snapshot and is not presented as the verified final PMFF/LFA implementation.

## Results reported in the manuscript

Mask AP is reported in percent. These are manuscript results, rather than measurements produced by the module smoke check.

| Dataset / validation split | Input size | YOLO11s-seg Mask AP | DRPMNet Mask AP | Gain (percentage points) |
|---|---|---:|---:|---:|
| COCO val2017 | 640 × 640 | 36.3 | 37.2 | +0.9 |
| Cityscapes val | 1024 × 1024 | 30.5 | 32.4 | +1.9 |
| BDD100K instance val | 1024 × 1024 | 28.7 | 30.3 | +1.6 |

| Platform / deployment setting | Input size | Mean end-to-end FPS | End-to-end P95 |
|---|---|---:|---:|
| Tesla P40, FP32, batch 1 | 640 × 640 | 77.7 | Not reported |
| Tesla P40, FP32, batch 1 | 1024 × 1024 | 27.3 | Not reported |
| Jetson Orin Nano Super 8 GB, 25 W, TensorRT FP16, batch 1 | 640 × 640 | 25.6 | 46.2 ms |

At 640 × 640, the manuscript reports 13.79 M deployment parameters and 48.1 GFLOPs. The Jetson FP16 configuration achieves 37.1 Mask AP on COCO val2017; its end-to-end P50/P95/P99 latencies are 39.1/46.2/52.2 ms. The 1024 × 1024 P40 setting and the listed Jetson setting fall below the 30 FPS average-throughput target; Jetson P95 also exceeds a 33.33 ms frame budget.

Timing uses five runs, each with 200 warm-up frames and 2,000 timed frames. Jetson end-to-end timing starts from decoded images in memory and includes preprocessing, transfer, engine execution, NMS, and mask decoding. Disk I/O, saving, and visualization are excluded. See [the manuscript protocol record](paper_docs/manuscript_protocol.md) for the recorded settings.

## Available files

| Path | Contents |
|---|---|
| `drpmnet/modules/drdbn.py` | Dual-branch routing module snapshot |
| `drpmnet/modules/pmaf.py` | Progressive fusion snapshot and auxiliary edge heads |
| `drpmnet/modules/eacl.py` | Boundary contrastive loss snapshot |
| `drpmnet_configs/` | COCO and Cityscapes dataset path/class templates |
| `scripts/smoke_test_drpmnet.py` | CPU forward/backward checks for standalone modules |
| `paper_docs/` | Release scope and manuscript settings |

## Check the standalone modules

The smoke-check environment is Python 3.12 with PyTorch 2.5.1 on CPU. This is separate from the manuscript's training environment. No Ultralytics installation, dataset, checkpoint, or GPU is required for these checks.

```bash
git clone https://github.com/Aeijou37/DRPMNet.git
cd DRPMNet
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Then install PyTorch and run the check. The CPU wheel command is intended for Windows/Linux:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python scripts/smoke_test_drpmnet.py
```

For another supported platform, install PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) before running the check. The standalone module dependency is recorded in `requirements.txt`.

Example module use:

```python
import torch
from drpmnet import DRDBN

block = DRDBN(in_channels=32).eval()
with torch.no_grad():
    features = block(torch.randn(1, 32, 16, 16))
print(features.shape)  # torch.Size([1, 32, 16, 16])
```

This checks a feature-processing block. It does not construct a full instance segmentation model or reproduce the paper's accuracy and speed tables.

## Datasets

Obtain the original datasets from their providers: [COCO](https://cocodataset.org/), [Cityscapes](https://www.cityscapes-dataset.com/), and [BDD100K](https://github.com/bdd100k/bdd100k). Dataset images and annotations are not redistributed here.

The example YAML files use relative roots under `datasets/`. Prepare the instance annotations separately, verify the polygon labels and class order, and set paths for your loader. Images and labels should be placed in matching split directories, for example `images/train2017` and `labels/train2017` for COCO. These templates do not convert native annotations. A BDD100K instance template is not included because the available source configuration's class mapping has not been reconciled with the manuscript evaluation.

## Manuscript and citation

**DRPMNet: Dynamic Routing and Progressive Multiscale Feature Fusion for Real-Time Instance Segmentation**

Guojie Li, Wenxia Bao, Wentao Guo, Jianwei Huang, Zhiyou Cheng.

No publication DOI or accepted-publication metadata is provided in this release. Until a stable paper record is available, this repository may be referenced directly:

```bibtex
@misc{li_drpmnet_code,
  author = {Li, Guojie and Bao, Wenxia and Guo, Wentao and Huang, Jianwei and Cheng, Zhiyou},
  title = {{DRPMNet}: Dynamic Routing and Progressive Multiscale Feature Fusion for Real-Time Instance Segmentation},
  howpublished = {GitHub repository},
  url = {https://github.com/Aeijou37/DRPMNet},
  note = {Partial research code release}
}
```

## Acknowledgements and contact

The manuscript uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) as its baseline. We thank the developers and dataset providers. Third-party software and data retain their respective licenses and terms. No new repository-wide license is assigned by this source snapshot.

For questions about this release, please use [GitHub Issues](https://github.com/Aeijou37/DRPMNet/issues).
