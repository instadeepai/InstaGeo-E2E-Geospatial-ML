---
title: '`InstaGeo`: Compute-Efficient Geospatial Machine Learning from Data to Deployment'
tags:
  - geospatial
  - earth observation
  - machine learning
  - remote sensing
  - foundation models
  - deployment
authors:
  - name: Ibrahim Salihu Yusuf
    orcid: 
    affiliation: 1
  - name: Iffanice Houndayi
    affiliation: 1
  - name: Rym Oualha
    affiliation: 1
  - name: Mohamed Aziz Cherif
    affiliation: 1
  - name: Kobby Panford-Quainoo
    orcid: 0000-0003-1835-2509
    affiliation: 1
  - name: Arnu Pretorius
    affiliation: 1
affiliations:
  - name: InstaDeep
    index: 1
date: 10 February 2026
bibliography: paper.bib
---

# Summary
`InstaGeo` is an open-source Python framework that helps users turn raw satellite imagery and in-situ observations
into practical machine learning models and interactive web-map applications. It is built around three modular
components: (1) automated dataset creation that converts raw satellite imagery such as Sentinel-1, Sentinel-2,
and Harmonised Landsat Sentinel (HLS) into model-ready formats, (2) fine-tuning and task-specific distillation
of geospatial foundation models (GFMs), and (3) browser-based inference with interactive visualisation.

By connecting these pieces, `InstaGeo` closes the gap between GFM research and real-world deployment.
Manages the entire workflow—from geolocated field observations to production-ready predictions—so users
can move from data to decisions without requiring users to integrate disparate tools or implement custom
pipelines.

# Statement of Need

Geospatial foundation models trained on open-access multispectral imagery have significantly improved performance
on many Earth observation (EO) tasks. However, two key bottlenecks still limit the extent to which they
are used in practice.

First, most published GFMs provide only model checkpoints and do not release the data pipelines needed to
convert raw satellite imagery into model-ready inputs [@cong2023satmae], [@xiong2024dofa], [@mendieta2023gfm], 
[@hong2024spectralgpt], [@szwarcman2025prithvi]. Practitioners are left to independently implement
STAC querying, temporal alignment, cloud masking, and label rasterisation. This process is time-consuming,
error-prone, and often becomes the main barrier to using GFMs in real applications.

Second, the standard adaptation workflow typically fine-tunes the full encoder, regardless of how simple or
complex the downstream task is. As a result, adapted models maintain their original size and computational cost,
even when lighter models would be sufficient. This makes deployment on resource-constrained infrastructure
unnecessarily expensive and can prevent models from being used operationally.

These issues create a divide. Domain experts in areas like agriculture or disaster response often have valuable
in-situ observations but lack the machine learning expertise needed to effectively use GFMs. Conversely, ML
practitioners may be comfortable with model adaptation but are less familiar with the practical challenges of
working with geospatial data.

`InstaGeo` addresses both bottlenecks by bringing dataset construction, model compression, and deployment
into a single, coherent, open-source workflow. It aims to enable collaboration between geo-experts and ML
practitioners on deployable geospatial ML systems.

# State of the Field

Several geospatial foundation models have already shown strong performance on standard benchmarks.
SatMAE [@cong2023satmae] introduced masked autoencoder pre-training for Sentinel-2 time series. DOFA [@xiong2024dofa] and GFM [@mendieta2023gfm] use
teacher–student training schemes to reduce pre-training costs. SpectralGPT [@hong2024spectralgpt] focuses on preserving richer
spectral information, and Prithvi-EO-2.0 [@szwarcman2025prithvi] achieves strong results on GEO-Bench [@lacoste2023geobench] through large-scale
HLS pre-training.

Despite these advances, existing projects generally do not provide open-source, end-to-end pipelines for turning
raw satellite tiles into training chips. They also tend to produce full-sized models after fine-tuning, even when
smaller models would be sufficient for downstream tasks.

TorchGeo [@stewart2025torchgeo] is a widely used library that offers geospatial datasets, samplers, transforms, and pre-trained
models. However, it does not cover the entire path from in-situ observations through dataset construction and
model adaptation to deployed predictions. `InstaGeo` complements existing tools by focusing on the full process
of adapting geospatial foundation models and deploying them in real-world systems.

# Software Design

`InstaGeo` follows a modular design with three loosely coupled components:  (i) a data pipeline that converts raw imagery into task-ready,
multi-temporal chips, (ii) a model component for fine-tuning or distilling geospatial foundation models (GFMs)
for downstream tasks, and (iii) an interactive web-map application for running inference and visualising
predictions in the browser. This design is illustrated in \autoref{fig:components_overview}.

These components communicate through standardised artefacts—GeoTIFFs for imagery and PyTorch checkpoints for models—so they can be
used independently or combined into a full pipeline.

![**Overview of InstaGeo.** (a) The data pipeline builds cloud-masked multi-temporal chips and labels. (b) Teacher–student distillation trains a lightweight model. (c) The application serves predictions as web-map tiles. \label{fig:components_overview}](figs/appComponents.pdf){#sylt width="100%"}

**Modularity over monolithic integration.** Users can adopt only the parts they need: geo-experts can build
datasets with the data pipeline, while ML practitioners can plug in existing datasets and focus on model
adaptation and evaluation.

**Data pipeline and *chip_creator*.** The *chip_creator* module turns geolocated labels into paired image/label
chips by querying STAC catalogues, selecting suitable acquisitions (e.g., low cloud cover) and exporting multi-temporal tensors (T, C, H, W) plus segmentation maps as GeoTIFFs. It supports HLS and Sentinel-1/2 and is
extensible to other STAC sources.

**Task-specific distillation over universal fine-tuning.** `InstaGeo` supports teacher–student distillation: a student
with the first N encoder layers is trained with task loss plus KL divergence to a frozen teacher, reducing inference
cost with minimal accuracy loss.

**STAC-first data access and browser-based deployment.** The imagery is accessed through STAC APIs (cloud-hosted COGs), reducing local storage but requiring network access during the construction of the dataset. For
deployment, predictions are served as map tiles (TiTiler) and explored in a browser (see \autoref{fig:app_component}).


![**Application component interface.** Users draw bounding boxes to define regions of interest and select model to run inference, monitor task progress, and view predictions as interactive map overlays. \label{fig:app_component}](figs/appInterface.pdf){#sylt width="100%"}

# Research Impact

`InstaGeo` has been validated across three published benchmarks: flood mapping, multi-temporal crop classification, and desert locust breeding ground prediction. A key requirement for the broad adoption of geospatial
foundation models is a data pipeline that can accurately reproduce the exact chip tensors used during training,
ensuring that the performance of the original model is preserved at inference time.

Using `InstaGeo`’s data pipeline, we reconstructed a replica dataset from scratch that matches the spatial,
temporal, and spectral specifications of each original study, and then re-trained the corresponding models
on the replica.
Across all tasks, replica models reproduced the original performance within ±2 percentage
points mean Intersection over Union (pp mIoU) (see \autoref{table:results})—differences minimal enough to attribute to
floating point precision—demonstrating that `InstaGeo` effectively replicates complex EO pipelines with minimal
performance degradation.

Due to the ease of use of the data pipeline, we curated a larger crop segmentation dataset, achieving a new
state-of-the-art mIoU of 60.65\%, an improvement of 12 pp over the previous baseline.

Task-specific distillation reduced model size by up to 8× while retaining comparable accuracy—for example,
compressing a 389M-parameter encoder to 46M parameters with less than 1 pp mIoU loss on locust prediction.

End-to-end, the framework reduces the data-to-deployment cycle to under nine hours on standard hardware,
enabling rapid iteration for time-sensitive applications such as emergency flood response. Full benchmark
results and reproduction scripts are available in the project [repository](https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML).

| Task                                 | Model                   | GFM            | mIoU (std) | Acc   | mF1 (std)   | ROC-AUC (std) |
|:-------------------------------------|:-----------------------|:----------------|:----------|:------|:-----------|:-------------|
| Flood Mapping                         | Baseline              | Prithvi-V1-100M | 88.3 (0.3)| -     | 97.3 (0.1) | -            |
|                                       | InstaGeo-Baseline     | Prithvi-V1-100M | 88.53     | 97.24 | 93.71      | 99.16        |
|                                       | InstaGeo-Replica (HLS)| Prithvi-V1-100M | 85.40     | 96.39 | 91.78      | 97.15        |
|                                       | InstaGeo-Replica (S2) | Prithvi-V1-100M | 87.80     | 97.07 | 93.26      | 97.61        |
|                                       |                       |                 |           |       |            |              |
| Multi-Temporal Crop Segmentation (US) | Baseline              | Prithvi-V1-100M | 42.70     | 60.7  | -          | -            |
|                                       | InstaGeo-Baseline     | Prithvi-V1-100M | 48.07     | 65.77 | 64.34      | 95.79        |
|                                       | InstaGeo-Replica      | Prithvi-V1-100M | 47.87     | 66.10 | 64.19      | 95.82        |
|                                       |                       |                 |           |       |            |              |
| Locust Breeding Ground Prediction     | Baseline              | Prithvi-V1-100M | -         | 83.03 | 81.53      | -            |
|                                       | InstaGeo-Baseline     | Prithvi-V1-100M | 71.51     | 83.39 | 83.39      | 86.74        |
|                                       | InstaGeo-Replica      | Prithvi-V1-100M | 73.30     | 84.60 | 84.60      | 88.66        |

:InstaGeo reproduces unpublished data pipelines. For each task, we report the performance of (i) the Baseline
performance reported in the corresponding study, (ii) the InstaGeo-Baseline model trained on the authors’ data but using
InstaGeo’s model component, and (iii) the InstaGeo-Replica model trained on a dataset reconstructed entirely with InstaGeo.
The flood mapping replica has a version derived from HLS and another derived from Sentinel-2. \label{table:results}

# AI usage disclosure

No generative AI tools were used to generate the scientific claims, experimental results, or evaluations described
in this paper. Generative AI tools have been used for minor language editing and formatting, and all text
was reviewed by the authors for accuracy and clarity.

# References