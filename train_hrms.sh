#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l v100=1,s_vmem=25G
#$ -pe def_slot 4

lr=1e-3
lr_VQ=1e-3
epoch=150
bad_limit=0
batch_size=128
margin=1

model_save_path="model_save_path_v7"
device="cuda:0"
CODE="code_classification/train.py"

kmer=6
model="Encoder"
model_backbone='CNN'
latent_channels=32
latent_space='Euclid'
gamma=1
strategy_emd='hrms'
strategy_samp='random'
strategy_rank='dist'
strategy_perp=5
strategy_gamm=1.0
strategy_hrms_alph=2
strategy_hrms_beta=50
strategy_hrms_gamm=1
min_dist=0.1
seed=123

species_fa='data/host.fasta'
species_fcgr_np="data/species_fcgr_np_${kmer}.npy"
species_fcgr_id="data/species_fcgr_id_${kmer}.json"
id2acc='data/host_accs.json'
scientific_names='data/scientific_names.json'
lineages='data/lineages.json'

module load /usr/local/package/modulefiles/python/3.12.0
python3 $CODE --model $model --model_backbone $model_backbone --model_dir $model_save_path --seed $seed --kmer $kmer --margin $margin \
    --device $device --lr $lr --lr_VQ $lr_VQ --epoch $epoch --batch_size $batch_size --latent_channels $latent_channels --latent_space $latent_space --gamma $gamma\
    --strategy_emd $strategy_emd --strategy_samp $strategy_samp --strategy_rank $strategy_rank --strategy_perp $strategy_perp \
    --strategy_gamm $strategy_gamm --strategy_hrms_alph $strategy_hrms_alph --strategy_hrms_beta $strategy_hrms_beta --strategy_hrms_gamm $strategy_hrms_gamm \
    --min_dist $min_dist --bad_limit $bad_limit --epoch_track --show_names\
    --species_fa $species_fa --species_fcgr_np $species_fcgr_np --species_fcgr_id $species_fcgr_id --id2acc $id2acc --scientific_names $scientific_names --lineages $lineages