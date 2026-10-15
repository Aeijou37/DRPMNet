# Manuscript protocol record

The following settings are transcribed from the current manuscript and supplementary material. This file is descriptive metadata, not a runnable training configuration. Full reproduction also requires the final integration, preprocessing and annotation conversion, exact splits, seeds, training commands, weights, and evaluation/deployment tooling.

## Data and optimization

| Dataset | Train images | Validation images | Input size | Epochs |
|---|---:|---:|---|---:|
| COCO 2017 | 118,287 | 5,000 | 640 × 640 | 100 |
| Cityscapes | 2,965 | 492 | 1024 × 1024 | 300 |
| BDD100K instance subset | 6,939 | 987 | 1024 × 1024 | 300 |

Optimizer: SGD; initial learning rate: 0.01; momentum: 0.937; weight decay: 0.0005; cosine learning-rate schedule; global batch size: 32. Accuracy tables report the mean of three independent runs unless otherwise stated.

DRDB uses eight convolution groups and context dilation 2. EACL uses loss weight 0.10, temperature 0.07, boundary threshold 0.50, a minimum of 10 candidate positions, a maximum of 200 sampled positions per image, and 128-dimensional projected representations. The manuscript constructs ground-truth boundaries using fixed 5 × 5 dilation/erosion, combines instance boundaries by pixelwise maximum, and maps them to the feature grid by adaptive max pooling. The strong view uses channel dropout with probability 0.10; the weak view is unperturbed and its projected target representations use stop-gradient. EACL auxiliary branches are excluded from the deployment graph.

## Recorded software

| Component | Manuscript record |
|---|---|
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |
| TensorRT | 8.6.0 |
| Ultralytics, training | 8.3.235 |
| Ultralytics, inference | 8.4.38 |

The standalone release check instead uses Python 3.12 and PyTorch 2.5.1 on CPU. That check is not a recreation of the manuscript environment.

## Deployment timing

Tesla P40: one GPU, FP32, batch 1, with preprocessing through postprocessing included in mean throughput. Jetson Orin Nano Super 8 GB: 25 W mode, TensorRT FP16, batch 1, 640 × 640. Five runs are measured per model, each with 200 warm-up and 2,000 timed frames.

Jetson end-to-end timing starts from decoded images in memory and includes resizing/normalization, transfer, engine execution, NMS, and mask decoding. Disk I/O, saving, and visualization are excluded. Engine-only timing is a different measurement and should not be substituted for end-to-end latency. Likewise, `1000 / mean FPS` is an equivalent mean frame time, not P95 latency.
