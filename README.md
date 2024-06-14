# DGCL

A contrastive learning method for predicting cancer driver genes based on graph diffusion

## Introduction

**DGCL** is a method for predicting cancer driver genes based on comparative learning under graph diffusion. Firstly, it obtains the corresponding gene-gene network from known protein-protein interaction relationships. Then, personalized PageRank is used for graph diffusion on this gene-gene network to obtain a diffusion-based gene-gene network. Next, comparative learning is performed on these two networks. During the comparative learning process, edge rounding and feature masking are applied to enhance the data of these two networks. The enhanced networks are input into a Chebyshev encoder with shared parameters for feature learning, utilizing neighborhood comparative learning loss as a constraint. Finally, the learned network features are further passed through a specific encoder for network-specific feature embedding learning. Both node classification and link prediction are used as constraints simultaneously. The final learned feature representations are concatenated and logistic regression classifiers are employed to learn the final feature representations in the two networks. The fusion of these features is utilized in a logistic regression model to predict cancer driver genes.

## Requirements

- Python 3.7
- Pytorch 1.9.1+cu111
- torch Geometric 2.0.4
- torch scatter 2.0.8
- torch sparse 0.6.11
- torch cluster 1.5.9
- torch spline conv 1.2.1
- pyyaml 6.0

## Data

| File Name          | Format           | Size        | Description                                                                                                    |
|:------------------:|:----------------:|:-----------:|:--------------------------------------------------------------------------------------------------------------:|
| `CPDB_data.pkl`    | --               | --          | This file contains the PPI network, gene features, gene names, and gene label information of the CPDB dataset. |
| `ppi.pkl`          | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of PPI network.                                                               |
| `ppi_selfloop.pkl` | torch.sparse_coo | 13627,13627 | Adjacency matrix (sparse matrix) of PPI network with self connection.                                          |
| `k_sets.pkl`       | dict             | --          | It preserves the data partitioning of the model during ten-fold cross-validation tests.                        |
| `Str_feature.pkl`  | tensor           | 13627,16    | This is a structural feature obtained through the Node2VEC algorithm on PPI network.                           |

## Running DGCL

Firstly,you should set the hyperparameter of the model through the configuration file config.yaml.

- `drop_edge_rate_1`:edge abandonment probability of Protein-protein network.

- `drop_edge_rate_2`:edge abandonment probability of  graph diffusion network.

- `drop_feature_rate_1`:feature masking probability of Protein-protein network.

- `drop_feature_rate_2`:feature masking probability of  graph diffusion network.

- `tau`:the neighborhood contrastive learning loss temperature hyperparameter, `[0,1].`

Then,you can run `python train_DGCL.py --dataset=CPDB --cancer_type=pan-cancer`

`--dataset`default is `CPDB` dataset,`--cancer_type`default is `pan-cancer`.

If you want to train a single cancer model, you can change the `cancer_type` for training, such as `python train_DGCL.py --cancer_type=brca`


