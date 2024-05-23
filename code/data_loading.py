# data loading for whole genome samples

import torch
import numpy as np
import json
import math
import os
import random
import itertools
from pyfaidx import Fasta
from collections import defaultdict
from torch.utils.data import Dataset

def my_collate_fn(batch):
    images, lineages = list(zip(*batch))
    imgs = torch.unsqueeze(torch.tensor(images, dtype = torch.float32), dim=1)
    lins = torch.tensor(lineages, dtype = torch.int32)
    return imgs, lins

class FastaData:
    def __init__(self, filenames, kmer, valid_ratio=0.1, test_ratio=0.1, local_norm=False, norm_type='standard'):
        species_fa = filenames['species_fa']
        species_fcgr_np = filenames['species_fcgr_np']
        species_fcgr_id = filenames['species_fcgr_id']
        acc2id_path = filenames['acc2id']
        id2acc_path = filenames['id2acc']
        scientific_names_path = filenames['scientific_names']
        lineages_path = filenames['lineages']

        # load meta infos
        with open(scientific_names_path, 'r') as f1, open(lineages_path, 'r') as f2:
            scientific_names = json.load(f1)
            lineages = json.load(f2)
        
        assert acc2id_path != '' or id2acc_path != ''
        if id2acc_path != '':
            with open (id2acc_path, 'r') as f:
                self.id2acc = json.load(f)
                self.acc2id = {}
                for taxid, accs in self.id2acc.items():
                    for acc in accs:
                        self.acc2id[acc] = taxid
        else:
            with open (id2acc_path, 'r') as f:
                self.acc2id = json.load(f)
                self.id2acc = defaultdict(list)
                for acc in self.acc2id.keys():
                    self.id2acc[self.acc2id[acc]].append(acc)

        # convert lineages into a valid format (e.g., None -> -1), but we do not accept such type as [None,...,None]
        lineages = {k: [taxid if isinstance(taxid, int) else -1 for taxid in v] for k,v in lineages.items()}
        self.hue_order = set()
        for lin in lineages.values():
            self.hue_order = self.hue_order | set(map(str, lin))
        self.hue_order = ['-1'] + list(self.hue_order - {'-1'})

        # load whole genomes and convert the genomes to FCGR images
        assert species_fa != '' or (species_fcgr_np != '' and species_fcgr_id != '')
        if species_fcgr_np == '' or species_fcgr_id == '':
            wgs = Fasta(species_fa)
            images = {}
            for taxid in self.id2acc.keys():
                seqs = [wgs[acc][:].seq for acc in self.id2acc[taxid] if acc in wgs.keys()]
                if len(seqs) > 0:
                    images[taxid] = self.get_FCGRimage(seqs, kmer)
            assert len(images) > 0
            #print({key: value if value == None else len(value) for key, value in images.items()})
            # define the order of species in the dataset
            taxids = list(images.keys())
            random.shuffle(taxids)
            #print(taxids)
            #print([images[taxid] for taxid in taxids])
            raw_X = np.array([images[taxid] for taxid in taxids], dtype=np.float32)
            savepath = os.path.join(os.path.dirname(species_fa), 'species_fcgr_')
            np.save(savepath + f'np_{kmer}.npy', raw_X)
            with open(savepath + f'id_{kmer}.json', 'w') as f:
                json.dump(taxids, f, indent=2)
        else:
            raw_X = np.load(species_fcgr_np)
            with open(species_fcgr_id, 'r') as f:
                taxids = json.load(f)
        valid_set = int((1-valid_ratio-test_ratio)*raw_X.shape[0])
        test_set = int((1-test_ratio)*raw_X.shape[0])
        # trn, val, tst split
        self.taxids_list = [taxids[:valid_set], taxids[valid_set:test_set], taxids[test_set:]]
        self.raw_X_list = [raw_X[:valid_set], raw_X[valid_set:test_set], raw_X[test_set:]]
        self.scientific_names_list = [[scientific_names[taxid] for taxid in taxids] for taxids in self.taxids_list]
        self.lineages_list = [[lineages[taxid] for taxid in taxids] for taxids in self.taxids_list]
        self.scientific_names_dic = scientific_names

        # data normalization
        if local_norm:
            axis = 0
        else:
            axis = None
        if norm_type == 'none':
            self.mn = np.zeros((1, 1, 1))
            self.sc = np.ones((1, 1, 1))
        elif norm_type == 'minmax':
            self.mn = self.raw_X_list[0].min(axis=axis, keepdims=True)
            self.sc = self.raw_X_list[0].max(axis=axis, keepdims=True) - self.mn
        elif norm_type == 'standard':
            self.mn = self.raw_X_list[0].mean(axis=axis, keepdims=True)
            self.sc = self.raw_X_list[0].std(axis=axis, keepdims=True)

    def __len__(self):
        return len(self.id2acc.keys())

    def get_hue_order(self):
        return self.hue_order

    def get_names_dic(self):
        return self.scientific_names_dic

    def get_dataset(self, i):
        return FastaDataset(self.raw_X_list[i], self.lineages_list[i], self.mn, self.sc)
    
    # code reference for count_kmers, probablities, and chaos_game_representation: 
    # https://towardsdatascience.com/chaos-game-representation-of-a-genetic-sequence-4681f1a67e14
    
    def count_kmers(self, sequences, k):
        d = defaultdict(int)
        for sequence in sequences:
            for i in range(len(sequence)-(k-1)):
                d[sequence[i:i+k]] += 1
            for key in list(d.keys()):
                if "N" in key:
                    del d[key]
        return d

    def probabilities(self, kmer_count, k):
        probabilities = defaultdict(float)
        total_counts = sum(value for value in kmer_count.values())
        for key, value in kmer_count.items():
            probabilities[key] = float(value) / total_counts
        return probabilities
            
    def chaos_game_representation(self, probabilities, k):
        array_size = int(math.sqrt(4**k))
        chaos = []
        for i in range(array_size):
            chaos.append([0]*array_size)

        maxx, maxy = array_size, array_size
        posx, posy = 1, 1

        for key, value in probabilities.items():
            for char in reversed(key):
                if char == "T" or char == "t":
                    posx += maxx / 2
                elif char == "C" or char == "c":
                    posy += maxy / 2
                elif char == "G" or char == "g":
                    posx += maxx / 2
                    posy += maxy / 2
                maxx = maxx / 2
                maxy /= 2

            chaos[int(posy)-1][int(posx)-1] = value
            maxx = array_size
            maxy = array_size
            posx = 1
            posy = 1

        return chaos
    
    def get_FCGRimage(self, sequences, k):
        kmer_count = self.count_kmers(sequences, k)
        probabilities = self.probabilities(kmer_count, k)
        return self.chaos_game_representation(probabilities, k)

class FastaDataset(Dataset):
    def __init__(self, raw_X, lineages, mn, sc):
        self.lineages = lineages
        self.mn = mn
        self.sc = sc
        self.X = self.norm(raw_X)

    def norm(self, X):
        return (X - self.mn) / self.sc

    def renorm(self, X):
        return X * self.sc + self.mn

    def get_lineages_unique(self, min_counts=2, return_counts=False, bottom_only=True): # for use in VQ module and CrossEntropyLoss
        lineages = np.array(self.lineages, dtype=np.int32)
        u, index, counts = np.unique(lineages[:,1:], return_index=True, return_counts=True, axis=0)
        if bottom_only:
            mask = (u[:,0] > 0) & (counts >= min_counts)
        else:
            mask = counts >= min_counts
        index = index[mask]
        if return_counts:
            return lineages[index], counts[mask]
        else:
            return lineages[index]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.lineages[idx])
