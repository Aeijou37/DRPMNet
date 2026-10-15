# Release status

This is a partial research source snapshot. It provides independently importable PyTorch modules and dataset path templates. It does not yet provide a verified implementation of the complete final manuscript.

## Included

- `DRDBN`, `PMAF`, and `EACL`, with their original class names and tensor computations retained.
- COCO and Cityscapes class/path examples, with local machine paths removed.
- Standalone module forward/backward checks.
- A record of the training and deployment settings reported in the manuscript.

Packaging removes private path comments, obsolete commented alternatives, and stale docstrings. The logging-only Ultralytics dependency is replaced by Python's standard `logging` module. Tensor operations, registered parameters, buffers, and default arguments are retained from the source snapshot, including unused legacy layers.

## Alignment with the manuscript

| Component | Current source snapshot | Final manuscript description / remaining work |
|---|---|---|
| DRDB | Two grouped branches and per-image softmax gates; the second convolution in each branch ends in BN, and the gate contains no BN | Reconcile normalization and activation placement with the final equations and diagram |
| PMFF | `PMAF` performs two-stage adjacent-scale fusion; the legacy `CrossScaleAttention` class currently runs concatenation followed by dense 1 × 1 and 3 × 3 convolutions | The manuscript's final LFA uses grouped and dilated grouped operations; the exact final implementation is not included |
| EACL | Contrastive objective accepts an external boundary map, with an optional learned edge-map fallback | Ground-truth morphology, boundary pooling, strong-view channel dropout, and final training integration are not included in this release |
| Model integration | Standalone modules only | Full model construction, feature wiring, criterion integration, training, evaluation, and export remain unavailable |
| BDD100K | No instance dataset template is included | Reconcile the available 19-class configuration with the instance annotations and manuscript evaluation mapping |

`EACL.initialize()` is a legacy warm-up helper with a hard-coded 256-channel input. It is not required by `EACL.forward()` and should only be used with the corresponding channel setting. The smoke check calls `forward()` directly. For manuscript-style sampling, callers must provide a valid boundary map aligned to the feature grid; using the fallback edge extractor alone does not implement that sampling procedure.

## Not included

Full training/validation scripts, annotation converters and exact split manifests, final integrated model configuration, trained checkpoints, TensorRT engines, deployment benchmarks, raw datasets, private logs, and manuscript drafts are not part of this release. No complete reproduction command is advertised.

Passing the module smoke check establishes basic shape, finite-loss, and gradient behavior in the tested CPU environment. It does not verify paper-level accuracy, deployment speed, full-model training, mixed precision, distributed training, or checkpoint compatibility.
