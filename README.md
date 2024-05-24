# Deep Metric Learning for Hierarchical Structured Data with Stochastic Neighbor Embedding

This repository is the official implementation of [Deep Metric Learning for Hierarchical Structured Data with Stochastic Neighbor Embedding](https://arxiv.org/abs/2030.12345). 

>![tSNE4HDML_figure.png](https://github.com/yoheyokubo/Images/blob/f33751e5a4c5f4910d15836a81e332f62ded444b/tSNE4HDML_figure.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Our scripts can be run on a single GPU. As for the dataset, we offer several options:
> -  If you do not care about the dataset and just want to run our scripts, please run script files (train_*.sh) because we uploaded preprocessed data.
> -  If you are interested in how the preprocessed data was created, please first download [host.fasta](https://zenodo.org/records/11274359) and locate it below our data directory. Next, erase parts related to *species_fcgr_np* in script files (train_*.sh).
> -  If you are interested in how the original dataset (host.fasta) and meta information (*.json) were created, please see our notebook in the data directory.

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

Pretrained models (tSNE or HRMS) are in the checkpoint directory.

## Results

Our model achieves the following performance:

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>![tSNE4HDML_table.png](https://github.com/yoheyokubo/Images/blob/f33751e5a4c5f4910d15836a81e332f62ded444b/tSNE4HDML_table.png) 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
