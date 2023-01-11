[PyTorch] Reconstruction by inpainting for visual anomaly detection (RIAD)
=====
PyTorch implementation of "Reconstruction by inpainting for visual anomaly detection (RIAD)"

## Concept
<div align="center">
  <img src="./figures/concept.png" width="400">
  <p>Concept ot the RIAD [1].</p>
</div>

### Model
<div align="center">
  <img src="./figures/model.png" width="750">
</div>

### Training Strategy

#### Overal Procedure
<div align="center">
  <img src="./figures/algo1.png" width="750">
</div>

#### Preprocessing
<div align="center">
  <img src="./figures/algo2.png" width="750">
</div>

#### Inference and Postprocessing
<div align="center">
  <img src="./figures/algo3.png" width="750">
</div>

## Results
(preparing)

## Requirements
* PyTorch 1.11.0

## Reference
[1] Vitjan Zavrtanik et al. <a href="https://www.sciencedirect.com/science/article/pii/S0031320320305094">"Reconstruction by inpainting for visual anomaly detection."</a> Pattern Recognition, vol. 112, 2021.
