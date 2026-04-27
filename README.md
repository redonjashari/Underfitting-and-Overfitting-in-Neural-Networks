# Overfitting and Underfitting Across Dataset Complexity
### A Comparative Study on Auto MPG and NYC Taxi

**Author:** Redon Jashari  
**Affiliation:** School of Computer Science and Engineering, Constructor University  
**Contact:** rjashari@constructor.university

---

## Overview

This repository contains the code and paper for an empirical comparative study of overfitting and underfitting in feedforward neural networks across two regression benchmarks of contrasting complexity:

| Dataset | Samples | Features | Signal | Noise |
|---|---|---|---|---|
| Auto MPG | 392 | 7 | Strong (linear) | Low |
| NYC Green Taxi (Jan 2023) | 15,000 | 7 | Weak | High (irreducible) |

The central finding is that **the capacity threshold at which a model overfits is not a fixed property of the architecture — it depends critically on dataset complexity, sample size, and the signal-to-noise ratio of the task.**

---

## Paper

The full paper is available at [`Overfitting_and_Underfitting_in_Neural_Networks.pdf`](Overfitting_and_Underfitting_in_Neural_Networks.pdf).

---

## Model Architectures

Four MLP architectures are trained and evaluated on each dataset:

| Model | Layers | Units/Layer | Parameters | L2 (λ) |
|---|---|---|---|---|
| Underfitting | 1 | 2 | ≈17 | 10⁻⁴ |
| Good Fit | 2 | 16 | ≈545 | 10⁻⁴ |
| Overfitting | 5 | 64 | ≈20k | 10⁻⁴ |
| Regularised | 5 | 64 | ≈20k | 0.1 |

All models use ReLU activation, Adam optimiser, and up to 5,000 training epochs. Input features are standardised to zero mean and unit variance using training-split statistics only.

---

## Evaluation Protocol

Each model is assessed by five protocols:

1. Single 80/20 train/test split — MSE and predicted-vs-actual scatter plots
2. Training loss curves (log scale) — convergence analysis
3. 10-fold cross-validation MSE (mean ± std)
4. Learning curves — training set size swept from 10% to 100% (5-fold CV)
5. Hyperparameter sensitivity — units per layer ∈ {2, 4, 8, 16, 32, 64, 128}, 2-layer network

Cross-dataset generalisation gaps are normalised by target variance for a fair comparison.

---

## Key Results

**Auto MPG**
- The overfitting model achieves the lowest train MSE (1.794) but the highest test MSE (9.345)
- Regularisation (λ = 0.1) closes the gap substantially: test MSE drops from 9.345 → 8.190
- Hyperparameter sensitivity shows a clear U-shaped test MSE curve with a minimum near 16 units/layer

**NYC Taxi**
- All absolute MSE values are much higher due to irreducible noise (trip duration cannot be fully predicted from available features)
- The overfitting signal is weaker in relative terms; the train–test gap is narrower
- The hyperparameter sensitivity curve is notably flat — no clear U-shaped optimum
- L2 regularisation provides modest improvement because the dominant error is irreducible noise, not variance

**Cross-Dataset Comparison**
- Sharp overfitting is characteristic of low-complexity, low-noise data (Auto MPG)
- On complex, noisy data (NYC Taxi), overfitting can be mistaken for irreducible noise if only aggregate metrics are examined

---

## Repository Structure

```
.
├── experiment.py                                   # Full reproducible experiment
├── Overfitting_and_Underfitting_in_Neural_Networks.pdf
├── Plots-and-Figures/                              # Generated output figures
└── README.md
```

### Generated Plots

Running `experiment.py` produces the following figures for each dataset (prefix `plot_mpg_` or `plot_taxi_`):

| File | Description |
|---|---|
| `*_eda.png` | Target distribution and feature correlation matrix |
| `*_feature_importance.png` | Absolute Pearson correlation with target |
| `*_predictions.png` | Predicted vs. actual scatter for all four models |
| `*_loss_curves.png` | Training loss curves (log scale) |
| `*_mse.png` | Train vs. test MSE bar chart |
| `*_crossval.png` | 10-fold CV MSE (mean ± std) |
| `*_learning_curves.png` | Learning curves (train size vs. MSE) |
| `*_hyperparam.png` | Hyperparameter sensitivity curve |
| `plot_comparison_gap.png` | Normalised generalisation gap — both datasets side by side |
| `plot_comparison_hyperparam.png` | Hyperparameter sensitivity — both datasets side by side |

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
pyarrow        # for reading the NYC Taxi .parquet file
```

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib pyarrow
```

---

## Reproducing the Experiment

```bash
python experiment.py
```

The script will:
1. Download the Auto MPG dataset from the UCI ML Repository
2. Download the NYC Green Taxi January 2023 dataset (~50 MB Parquet file)
3. Train all four models on both datasets
4. Generate and save all figures to the working directory
5. Print MSE tables to stdout

> The NYC Taxi data is sampled to 15,000 records for computational tractability (`N_SAMPLE = 15000` in `experiment.py`). Increase this value for more thorough results at the cost of longer training time.

---

## References

1. Goodfellow et al., *Deep Learning*, MIT Press, 2016
2. Geman & Bienenstock, "Neural networks and the bias/variance dilemma," *Neural Computation*, 1992
3. Hastie et al., *The Elements of Statistical Learning*, Springer, 2009
4. Hornik et al., "Multilayer feedforward networks are universal approximators," *Neural Networks*, 1989
5. Zhang et al., "Understanding deep learning (still) requires rethinking generalisation," *CACM*, 2021
6. Belkin et al., "Reconciling modern machine-learning practice and the classical bias–variance trade-off," *PNAS*, 2019
7. Kingma & Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015
8. Srivastava et al., "Dropout," *JMLR*, 2014
9. Quinlan, "Combining instance-based and model-based learning," *ICML*, 1993
10. Dua & Graff, "UCI Machine Learning Repository," 2019
11. NYC TLC, "TLC Trip Record Data," 2023
16. Bashir et al., "An information-theoretic perspective on overfitting and underfitting," *arXiv:2010.06076*, 2020
