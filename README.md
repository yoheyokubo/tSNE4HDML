# Deep Metric Learning for Hierarchical Structured Data with Stochastic Neighbor Embedding

This repository is the official implementation of [Deep Metric Learning for Hierarchical Structured Data with Stochastic Neighbor Embedding](https://arxiv.org/abs/2030.12345). 

>![tSNE4HDML_figure.png](https://github.com/yoheyokubo/Images/blob/f33751e5a4c5f4910d15836a81e332f62ded444b/tSNE4HDML_figure.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Our scripts require a single GPU. As for the dataset, we offer several options depending on you interests:
-  If you do not care about the dataset and just want to run our scripts, please run script files (train_*.sh) in the following sections because we uploaded preprocessed data in our data directory.
-  If you are interested in how the preprocessed data was produced, please first download [host.fasta](https://zenodo.org/records/11276021) and locate it below the data directory. Next, erase parts related to the preprocessed data *species_fcgr_np* in script files (train_*.sh).
-  If you are interested in how the original dataset (host.fasta) and meta information (*.json) were created, please see our notebook in the data directory.

## Training

To train the model(s) in the paper, run this command for our method:

```train
bash train_tsne.sh
```
for HRMS:

```train
bash train_hrms.sh
```

## Evaluation

To evaluate trained models, run:

```eval
bash eval.sh
```
Please change the hyperparameter _model_checkpoint_ in the file according to which model you want to test: our method (tSNE) or HRMS.

## Pre-trained Models

Pretrained models (tSNE or HRMS) are included in the checkpoint directory.

## Results

Our model achieves the following performance:

>![tSNE4HDML_table.png](https://github.com/yoheyokubo/Images/blob/f33751e5a4c5f4910d15836a81e332f62ded444b/tSNE4HDML_table.png) 


## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
