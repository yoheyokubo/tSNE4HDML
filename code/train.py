# train the constrastive learning
import argparse, time, random

from data_loading import FastaData, my_collate_fn
from torch.utils.data import DataLoader
from model import CNN, VAE, VQ, DivergenceLoss, EmbedDistanceLoss
from eval import get_eval_metrics

import torch
import multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import os
import numpy as np
import math
import wandb
import copy
import pandas as pd

wandb.login()

import warnings
warnings.filterwarnings("ignore")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    torch.cuda.manual_seed_all(s)
    #add additional seed
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms = True

class Runner:
    def __init__(self, args):
        data_set = {
          'species_fa': args.species_fa,
          'species_fcgr_np': args.species_fcgr_np,
          'species_fcgr_id': args.species_fcgr_id,
          'acc2id': args.acc2id,
          'id2acc': args.id2acc,
          'scientific_names': args.scientific_names, 
          'lineages': args.lineages
        }
        set_seed(args.seed)
        # model train
        self.train(args.model, args.model_backbone, data_set, args.model_dir, args.device, args.kmer, args.margin, args.lr, args.lr_VQ, args.epoch, args.batch_size, args.gamma,\
            args.workers, args.latent_channels, args.latent_space, args.curvature, args.strategy_emd, args.strategy_samp, args.strategy_rank, \
            args.strategy_perp, args.strategy_gamm, args.strategy_hrms_alph, args.strategy_hrms_beta, args.strategy_hrms_gamm, \
            args.min_dist, args.epoch_track, args.show_names, args.bad_limit, args.seed)        

    def train(self, dl_model, model_backbone, data_set, model_dir, device, kmer, margin, lr, lr_VQ, n_epochs, batch_size, gamma,\
            num_workers, latent_channels, latent_space, c, strategy_emd, strategy_samp, strategy_rank, strategy_perp, strategy_gamm,\
            strategy_hrms_alph, strategy_hrms_beta, strategy_hrms_gamm, min_dist, \
            epoch_track, show_names, bad_limit, seed, verbose=True):

        # initialize config of wandb
        hyp_info = '{}-{}-{}-{}-{}-{}-{}-lr{}-lrVQ{}-perp{}-g{}-ha{}-hb{}-hg{}-min{}-seed{}'.format(dl_model, kmer, latent_space, latent_channels, strategy_emd, str(batch_size), gamma, str(lr), str(lr_VQ), strategy_perp, strategy_gamm, strategy_hrms_alph, strategy_hrms_beta, strategy_hrms_gamm, min_dist, seed)
        run = wandb.init(
            project='taxonomic-classification-p3',
            name=hyp_info,
            config={
                "model_structure": dl_model,
                "kmer": kmer,
                "batch_size": batch_size,
                "gamma": gamma,
                "learning_rate": lr,
                "learning_rate_VQ": lr_VQ,
                "epoch": n_epochs,
                "latent_channels": latent_channels,
                "latent_space": latent_space,
                "strategy_emd": strategy_emd,
                "strategy_samp": strategy_samp,
                "strategy_rank": strategy_rank,
                "strategy_perp": strategy_perp,
                "strategy_gamm": strategy_gamm,
                "umap_min_dist": min_dist,
                "seed": seed
            },
        )

        cores = mp.cpu_count()
        if num_workers > 0 and num_workers < cores:
            cores = num_workers
        model_info = hyp_info + '.pth'
        self.model_path = os.path.join(model_dir, model_info)
        print(" |- Model path: " + self.model_path)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.device = device
        self.dl_model = dl_model
        self.latent_space = latent_space
        self.epoch_track = epoch_track
        self.gamma = gamma

        """data loading phase"""
        if verbose:
            print(" |- Start preparing dataset...")

        start_dataload = time.time()
        data = FastaData(data_set, kmer)
        trn_dataset, val_dataset, tst_dataset = data.get_dataset(0), data.get_dataset(1), data.get_dataset(2)
        self.dataloaders = [DataLoader(trn_dataset, shuffle=True, batch_size=min(len(trn_dataset), batch_size), collate_fn=my_collate_fn, num_workers=num_workers),
                            DataLoader(val_dataset, batch_size=min(len(val_dataset), batch_size), collate_fn=my_collate_fn, num_workers=num_workers),
                            DataLoader(tst_dataset, batch_size=min(len(tst_dataset), batch_size), collate_fn=my_collate_fn, num_workers=num_workers)]
        self.hue_order = data.get_hue_order()
        self.names_dic = data.get_names_dic() if show_names else None
        self.df_trn = None

        print(" |- loading [ok].")
        used_dataload = time.time() - start_dataload
        print("  |-@ used time:", round(used_dataload,2), "s")

        start_train = time.time()

        """model loading phase"""
        if "Encoder" in dl_model and model_backbone == 'CNN':
            model = CNN(kmer, latent_channels = latent_channels, latent_space = latent_space).to(device)
        elif 'VAE' in dl_model and model_backbone == 'CNN':
            model = VAE(kmer, latent_channels = latent_channels, latent_space = latent_space).to(device)
            self.criterion_rec = nn.MSELoss()
            self.criterion_div = DivergenceLoss(latent_space=latent_space, c=c) #alpha=1, gamma=int(((2**kmer)**2)/latent_channels
        self.criterion_emd = EmbedDistanceLoss(batch_size, latent_space=latent_space, strategy_emd=strategy_emd, strategy_samp=strategy_samp, strategy_rank=strategy_rank,
                                               strategy_perp=strategy_perp, strategy_gamm=strategy_gamm, strategy_hrms_alph=strategy_hrms_alph, strategy_hrms_beta=strategy_hrms_beta, 
                                               strategy_hrms_gamm=strategy_hrms_gamm, min_dist=min_dist, 
                                               lin_trn_counts=trn_dataset.get_lineages_unique(min_counts=1, return_counts=True, bottom_only=False) if strategy_emd=='ce' else None,
                                               n_channels=latent_channels)
        if strategy_emd in ['ce', 'tsneL']:
            self.criterion_emd = self.criterion_emd.to(device)
        model_VQ = VQ(trn_dataset.get_lineages_unique(), embedding_dim = latent_channels, latent_space=latent_space).to(device) if 'VQ' in dl_model else None
        
        #parameters = list(model.parameters()) + (list(model_VQ.parameters()) if model_VQ is not None else [])
        optimizer_args = [{'params': model.parameters(), 'lr': lr}]
        n_param = sum([p.nelement() for p in model.parameters()])
        if 'VQ' in dl_model:
            optimizer_args += [{'params': model_VQ.parameters(), 'lr': lr_VQ}]
            n_param += sum([p.nelement() for p in model_VQ.parameters()])
        if strategy_emd in ['ce', 'tsneL']:
            optimizer_args += [{'params': self.criterion_emd.parameters(), 'lr': lr}]
            n_param += sum([p.nelement() for p in self.criterion_emd.parameters()])
        self.optimizer = optim.Adam(optimizer_args)
        self.model = model
        self.model_VQ = model_VQ

        if verbose:
            print(f" |- Total number of {dl_model} has parameters %d:" %(n_param))
            print("  |- Training started ...")

        """training phase"""
        n_digits = len(str(n_epochs+1))
        bad_counts = 0
        bst_val_score = 0
        bst_model = self.model
        self.X_trn_saved = None
        self.X_val_saved = None
        self.X_tst_saved = None
        self.lin_trn_saved = None
        self.lin_val_saved = None
        self.lin_tst_saved = None
        for epoch in range(1, n_epochs+1):
            
            #if epoch_track:
            #    file_path = model_path + f'-{{:0={n_digits}}}-out.csv'.format(ep)
            #    dp = calc_embeddings(task, dl_model, model_ph, model_bt, train_p_dataloader, host_dataloader, l2s_dic, device, file_path, df_host_tree=df_host_tree, palette=cmap_categorical)
            #    print(dp)
            #    wandb.log(dp)

            trn_results = self.one_epoch(epoch, 0)
            epoch_info = f'epoch = {{:0={n_digits}}} , trn loss = {{:.6f}}'.format(epoch, trn_results['loss'])

            with torch.no_grad():
                val_results = self.one_epoch(epoch, 1)
                epoch_info += ' , val loss = {:.6f}'.format(val_results['loss'])

                tst_results = self.one_epoch(epoch, 2)
                epoch_info += ' , tst loss = {:.6f}'.format(tst_results['loss'])

            print(epoch_info)

            wandb.log({'trn_'+key:value for key, value in trn_results.items()})
            wandb.log({'val_'+key:value for key, value in val_results.items()})
            wandb.log({'tst_'+key:value for key, value in tst_results.items()})

            if val_results['map'] > bst_val_score:
                bst_val_score = val_results['map']
                bad_counts = 0
                torch.save(self.model.state_dict(), self.model_path)
                with open(os.path.splitext(self.model_path)[0] + f'-X_trn-{epoch}.npy', 'wb') as f:
                    np.save(f, self.X_trn_saved.numpy().copy())
                with open(os.path.splitext(self.model_path)[0] + f'-lin_trn-{epoch}.npy', 'wb') as f:
                    np.save(f, self.lin_trn_saved.numpy().copy())
                with open(os.path.splitext(self.model_path)[0] + f'-X_val-{epoch}.npy', 'wb') as f:
                    np.save(f, self.X_val_saved.numpy().copy())
                with open(os.path.splitext(self.model_path)[0] + f'-lin_val-{epoch}.npy', 'wb') as f:
                    np.save(f, self.lin_val_saved.numpy().copy())
                with open(os.path.splitext(self.model_path)[0] + f'-X_tst-{epoch}.npy', 'wb') as f:
                    np.save(f, self.X_tst_saved.numpy().copy())
                with open(os.path.splitext(self.model_path)[0] + f'-lin_tst-{epoch}.npy', 'wb') as f:
                    np.save(f, self.lin_tst_saved.numpy().copy())
                print(list(tst_results.items()))
                dic = dict(**{'tst_' + k: v for k, v in tst_results.items() if isinstance(v, float)},
                           **{'val_' + k: v for k, v in val_results.items() if isinstance(v, float)},
                           **{'trn_' + k: v for k, v in trn_results.items() if isinstance(v, float)})
                dic.update(seed=seed, 
                           strategy_emd=strategy_emd,
                           strategy_perp=strategy_perp,
                           strategy_gamm=strategy_gamm,
                           strategy_hrms_alph=strategy_hrms_alph,
                           strategy_hrms_beta=strategy_hrms_beta,
                           strategy_hrms_gamm=strategy_hrms_gamm)
                df = pd.DataFrame.from_dict(dic, orient="index")
                df.to_csv(os.path.splitext(self.model_path)[0]+"-data.csv")
            else:
                bad_counts += 1
            if bad_limit > 0 and bad_counts >= bad_limit:
                break

        used_train = time.time() - start_train
        print(" @ used training time:", round(used_train,2), "s. Total time:", round(used_train+used_dataload,2))

    def one_epoch(self, epoch, mode):
        if mode == 0:
            self.model.train()
            self.X_trn = None
            self.lin_trn = None
        else:
            self.model.eval()
        if epoch == 1:
            if mode == 0:
                self.X_imgs_trn = None
            X_imgs = None
        X = None
        lin = None

        save_fig_path = os.path.splitext(self.model_path)[0] + '-{}-{}.png'.format(['trn', 'val', 'tst'][mode], epoch) if self.epoch_track else None

        trn_results={}
        trn_batch_count = 0
        
        #codebook, _ = self.model_VQ.get_codebook_infos()
        #print(codebook[:10])

        epoch_loss_emd, epoch_loss_rec, epoch_loss_div, epoch_loss_code = 0, 0, 0, 0
        with torch.autograd.set_grad_enabled(mode==0):
            for imgs, lineages in self.dataloaders[mode]:
                imgs = imgs.to(self.device)
                lineages = lineages.to(self.device)
                outputs = self.model(imgs)
                loss = 0
                c = None
                if 'Encoder' in self.dl_model:
                    embed_mu = outputs
                    #print(embed_mu[:10,-1,0])
                    #print(embed_mu[:10,-1,1])
                elif 'VAE' in self.dl_model:
                    rec, embed, embed_mu, embed_logvar = outputs
                    reconstruction_loss = self.criterion_rec(imgs, rec) 
                    divergence_loss = self.criterion_div(embed, embed_mu, embed_logvar)
                    #print(torch.isnan(reconstruction_loss).any())
                    #print(torch.isnan(divergence_loss).any())
                    epoch_loss_rec += reconstruction_loss.item()
                    epoch_loss_div += divergence_loss.item()
                    loss += reconstruction_loss + divergence_loss
                    c = torch.exp(embed_logvar[...,1]) if self.latent_space == 'GaussianManifoldL' else None
              
                embed_all = embed_mu
                lineages_all = lineages
                if 'VQ' in self.dl_model:
                    codebook_loss,_ = self.model_VQ(embed_mu, lineages)
                    epoch_loss_code += codebook_loss.item()
                    loss += codebook_loss
                    embed_code, lineages_code = self.model_VQ.get_codebook_infos()
                    embed_all = torch.cat([embed_mu, embed_code], dim=0)
                    lineages_all = torch.cat([lineages, lineages_code], dim=0)
               
                embed_distance_loss = self.gamma * self.criterion_emd(embed_all, lineages_all, c=c)
                epoch_loss_emd += embed_distance_loss.item()
                loss += embed_distance_loss

                if epoch == 1 and trn_batch_count <= 1:
                    print('input: ', torch.mean(imgs), torch.sum(imgs))
                    for param_name, param in self.model.named_parameters():
                        print(param_name, torch.mean(param))
                    print('output: ', torch.mean(embed_mu))


                if mode == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                X = embed_mu.detach().cpu() if X is None else torch.cat((X,embed_mu.detach().cpu()), 0)
                lin = lineages.detach().cpu() if lin is None else torch.cat((lin,lineages.detach().cpu()), 0)
                
                if epoch == 1:
                    X_imgs = imgs.detach().cpu() if X_imgs is None else torch.cat((X_imgs, imgs.detach().cpu()), 0)
                    if mode == 0 and trn_batch_count == 0:
                        trn_save_fig_path = os.path.splitext(save_fig_path)[0] + f'-minibatch{trn_batch_count}.png'
                        trn_results,_ = get_eval_metrics(X, lin, self.latent_space, epoch, save_fig_path=trn_save_fig_path, X_trn_emd = self.X_trn, lineages_trn = self.lin_trn)
                        trn_results = {k + f'-minibatch{trn_batch_count}': v for k,v in trn_results.items()}
                    trn_batch_count += 1

        pre_results = {}
        if epoch == 1:
            #print('ok')
            save_fig_path_pre = os.path.splitext(save_fig_path)[0] + '-wo_enc.png'
            X_emd_pre = X_imgs.reshape((X_imgs.shape[0], -1))
            X_trn_emd_pre = self.X_imgs_trn.reshape((self.X_imgs_trn.shape[0], -1)) if self.X_imgs_trn is not None else self.X_imgs_trn
            #print(self.X_imgs_trn == None, self.lin_trn == None)
            #print(X_emd_pre.shape)
            #if X_trn_emd_pre is not None:
            #    print('trn: ',  X_trn_emd_pre.shape)
            #else:
            #    print('trn: None')
            pre_results, df = get_eval_metrics(X_emd_pre, lin, 'Euclid', epoch, save_fig_path=save_fig_path_pre, X_trn_emd = X_trn_emd_pre, lineages_trn = self.lin_trn, names_dic=self.names_dic, hue_order=self.hue_order, wo_enc=True, df_trn=self.df_trn)
            self.df_trn = df if self.df_trn is None else self.df_trn
            pre_results = {k+'_pre-result':v for k,v in pre_results.items()}
            #print(pre_results)

        if mode == 0 and epoch % 10 == 0:
            with open(os.path.splitext(self.model_path)[0] + f'-X-{epoch}.npy', 'wb') as f:
                np.save(f, X.numpy().copy())
            with open(os.path.splitext(self.model_path)[0] + f'-lin-{epoch}.npy', 'wb') as f:
                np.save(f, lin.numpy().copy())

        if 'VQ' in self.dl_model:
            embed_code, lineages_code = self.model_VQ.get_codebook_infos()
            X_vis = torch.cat([X, embed_code.detach().cpu()], dim=0)
            #print(lin.dtype)
            #print(lineages_code.dtype)
            #print(lineages_code.cpu().dtype)
            lin_vis = torch.cat([lin, lineages_code.detach().cpu()], dim=0)
        else:
            X_vis = lin_vis = None

        results, _ = get_eval_metrics(X, lin, self.latent_space, epoch, save_fig_path=save_fig_path, X_trn_emd = self.X_trn, lineages_trn = self.lin_trn, names_dic=self.names_dic, hue_order=self.hue_order, X_vis=X_vis, lin_vis=lin_vis)

        results['loss_emd']  = epoch_loss_emd / len(self.dataloaders[mode].dataset)
        results['loss_rec']  = epoch_loss_rec / len(self.dataloaders[mode].dataset)
        results['loss_div']  = epoch_loss_div / len(self.dataloaders[mode].dataset)
        results['loss_code'] = epoch_loss_code/ len(self.dataloaders[mode].dataset)
        results['loss'] = (epoch_loss_emd + epoch_loss_rec + epoch_loss_div + epoch_loss_code) / len(self.dataloaders[mode].dataset)
        mavep_l = [v for k,v in results.items() if 'map' in k and 'max' not in k and 'heat' not in k]
        results['map'] = sum(mavep_l)/max(len(mavep_l),1)

        if mode == 0:
            self.X_trn = X
            self.lin_trn = lin
            if epoch == 1:
                self.X_imgs_trn = X_imgs
        if mode == 0:
            self.X_trn_saved = X
            self.lin_trn_saved = lin
        elif mode == 1:
            self.X_val_saved = X
            self.lin_val_saved = lin
        elif mode == 2:
            self.X_tst_saved = X
            self.lin_tst_saved = lin

        return {**results, **pre_results, **trn_results}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='<Representation learning for the whole genome taxonomic classification>')

    parser.add_argument('--model',  default="Encoder", type=str, required=True, choices=['Encoder', 'VAE', 'VQEncoder', 'VQVAE'], help='encoding model architecture')
    parser.add_argument('--model_backbone',  default="CNN", type=str, required=True, choices=['CNN'], help='encoding backbone')
    parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
    parser.add_argument('--device', default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
    parser.add_argument('--seed',   default=123,       type=int, required=False, help='seed for repetition')

    parser.add_argument('--kmer',       default=5,       type=int, required=True, help='kmer length')
    parser.add_argument('--margin',     default=1,       type=int, required=True, help='Margins used in the contrastive training')
    parser.add_argument('--lr',         default=1e-3,   type=float, required=False, help='Learning rate')
    parser.add_argument('--lr_VQ',         default=1e-3,   type=float, required=False, help='Learning rate for VQ model')
    parser.add_argument('--epoch',      default=20,       type=int, required=False, help='Training epcohs')
    parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
    parser.add_argument('--workers',     default=48,       type=int, required=False, help='number of worker for data loading')
    parser.add_argument('--latent_channels', default=512, type=int, required=False, help='the size of hidden dimension (+1 if hyperbolic)')
    parser.add_argument('--latent_space', default='Euclid', type=str, required=False, choices=['GaussianManifold', 'Lorentz', 'Euclid'], help='latent space to embed data points into')
    parser.add_argument('--curvature', default=1.0, type=float, required=False, help='positive value for the constant negative curvature in Gaussian Manifold')
    parser.add_argument('--gamma', default=1.0, type=float, required=False, help='gamma hyperparameter for weighting embedding-distance-loss')
    parser.add_argument('--strategy_emd', default='slr', type=str, required=False, choices=['none', 'direct', 'slr', 'rrl', 'tsne', 'umap', 'ce', 'cl', 'dp', 'hrms', 'sneht', 'tsneL', 'tsneS', 'tsneB', 'tsneR', 'tsneC', 'tsneO', 'tsneT', 'tsneF', 'sne', 'hrmsDP', 'hrmsDN', 'hrmsDPDN', 'hrmsW', 'hrmsWDPDN', 'hrmsS', 'hrmsA', 'hrmsFTM', 'hrmstsne'], help='strategy to embed real distance into latent space')
    parser.add_argument('--strategy_samp', default='random', type=str, required=False, choices=['random', 'anchor'], help='strategy to sample pairs when using slr')
    parser.add_argument('--strategy_rank', default='rank', type=str, required=False, choices=['rank', 'dist'], help='strategy to rank pairs when using slr')
    parser.add_argument('--strategy_perp', default=5.0, type=float, required=False, help='perplexity when using tsne and umap')
    parser.add_argument('--strategy_gamm', default=0.1, type=float, required=False, help='gamma hyperparameter for latent Cauchy distribution when using tsne and umap (the smaller the peakier)')
    parser.add_argument('--strategy_hrms_alph', default=2.0, type=float, required=False, help='alpha when using hrms')
    parser.add_argument('--strategy_hrms_beta', default=2.0, type=float, required=False, help='beta when using hrms')
    parser.add_argument('--strategy_hrms_gamm', default=0.5, type=float, required=False, help='gamma when using hrms')
    parser.add_argument('--min_dist', default=0.1, type=float, required=False, help='min_dist when using umap')

    parser.add_argument('--bad_limit', default=25, type=int, required=False, help='number of epochs to endure for early stopping')
    parser.add_argument('--epoch_track', action='store_true', help="whether or not track epoch via embeddings")
    parser.add_argument('--show_names', action='store_true', help="whether or not show names of groups when visualizing")

    # data related input
    parser.add_argument('--species_fa',   default="",  type=str, required=False, help='Species fasta files')
    parser.add_argument('--species_fcgr_np',   default="",  type=str, required=False, help='Species FCGR images preprocessed and saved either in .npy or .npz format')
    parser.add_argument('--species_fcgr_id',   default="",  type=str, required=False, help='Species FCGR images preprocessed and saved in .josn format')
    parser.add_argument('--acc2id',   default="",  type=str, required=False, help='Accession number to TaxID dictionary')
    parser.add_argument('--id2acc',   default="",  type=str, required=False, help='TaxID to list of accession numbers dictionary')
    parser.add_argument('--scientific_names', default="",  type=str, required=True, help='TaxID to scientific name dictionary')
    parser.add_argument('--lineages', default="",  type=str, required=True, help='TaxID to lineage list dictionary')


    args = parser.parse_args()
    runner = Runner(args)
    #runner.run()
