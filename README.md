# **LCMR-3S: Learning Cross-modality Representation via Selective State Space Model for Depression Detection on Social Media [******** 2025]

[Quang Vinh Nguyen](https://github.com/vinhhust2806), 
Thanh Dong Nguyen,
Duc Duy Nguyen,
Doan Khai Ta,
Ji-Eun Shin,
Seung-Won Kim,
Hyung-Jeong Yang,
Soo-Hyung Kim

Official PyTorch implementation

# :fire: News
* **(September 16, 2024)**
  * Paper submitted at ****** 2025 (Rank A) ! ⌚

<hr />

## 💿 Installation

```python
pip install -r requirements.txt
```

## 🏁 Dataset Preparation

The Twitter dataset could be dowloaded [here](https://drive.google.com/open?id=11ye00sHFY5re2NOBRKreg-tVbDNrc7Xd).

Please contact the respective authors in above referenced paper for accessing the Reddit dataset.

Uban, Ana-Sabina, Berta Chulvi, and Paolo Rosso. [Explainability of Depression Detection on Social Media: From Deep Learning Models to Psychological Interpretations and Multimodality](https://link.springer.com/chapter/10.1007/978-3-031-04431-1_13). In Early Detection of Mental Health Disorders by Social Media Monitoring, pp. 289-320. Springer, Cham, 2022.

Dataset preparation is defined in bash script in `scripts/extract_embeddings.sh`.

## 🚀 Training and Evaluating

Experiments are defined in the bash script in `experiments/run_experiments.sh`.
