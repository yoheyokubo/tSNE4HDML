#!/bin/bash

batch_size=128
device="cuda:0"
CODE="code/eval.py"
model_checkpoint="checkpoint/model-Encoder-space-Euclid-epoch-150-dim-32-lr-0.001-margin-1-emd-hrms-samp-random-rank-dist-seed-123.pth"

kmer=6
model="Encoder"
model_backbone='CNN'
latent_channels=32
latent_space='Euclid'

species_fa='data/host.fasta'
species_fcgr_np="data/species_fcgr_np_${kmer}.npy"
species_fcgr_id="data/species_fcgr_id_${kmer}.json"
id2acc='data/host_accs.json'
scientific_names='data/scientific_names.json'
lineages='data/lineages.json'

python3 $CODE --model $model --model_backbone $model_backbone --model_checkpoint $model_checkpoint --kmer $kmer \
    --device $device --batch_size $batch_size --latent_channels $latent_channels --latent_space $latent_space \
    --species_fa $species_fa --species_fcgr_np $species_fcgr_np --species_fcgr_id $species_fcgr_id --id2acc $id2acc --scientific_names $scientific_names --lineages $lineages
