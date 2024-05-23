# Deep Metric Learning for Hierarchical Structured Data with Stochastic Neighbor Embedding

This repository is the official implementation of [Deep Metric Learning for Hierarchical Structured Data with Stochastic Neighbor Embedding](https://arxiv.org/abs/2030.12345). 

>![tSNE4HDML_figure.png](https://github.com/yoheyokubo/Images/blob/f33751e5a4c5f4910d15836a81e332f62ded444b/tSNE4HDML_figure.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

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

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
