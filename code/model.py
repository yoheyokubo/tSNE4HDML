# preparing for the constrastive learning

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import log_softmax
from functools import partial
from scipy.optimize import curve_fit
import random
import itertools
import math
import numpy as np
from scipy.cluster.hierarchy import linkage
from collections import defaultdict

def embedding_distance(x0, x1, latent_space= 'Euclid', c=None):
    if latent_space == 'Euclid':
        d = torch.sqrt(torch.clamp(torch.sum((x0 - x1) ** 2, -1), min=1e-6))
    elif latent_space == 'Lorentz':
        d = geodesic_distance(x0, x1) #if train else geodesic_distance_poincare(x0, x1)
    elif latent_space == 'GaussianManifold' or latent_space == 'GaussianManifoldL':
        #print(c)
        #print(x0.shape)
        #print(x1.shape)
        c_sqr = 1 if c is None else (torch.sqrt(c).to(x0) if torch.is_tensor(c) else torch.sqrt(torch.tensor(c)).to(x0))
        d = c_sqr * geodesic_distance_poincare_half(x0[...,0]/c_sqr, torch.exp(x0[...,1]/2), x1[...,0]/c_sqr, torch.exp(x1[...,1]/2))
    return d

def lineage_distance(lin1, lin2): # assume None is replaced by negative int
    d = 2 * torch.argmax(((lin1 >= 0) & (lin2 >= 0) * ((lin1 - lin2) == 0)).to(torch.int32), axis=-1)
    return d

def get_leaves_dict(z):
    # dictonary: each node (internal/external) to all the descendant leaves
    n = len(z)+1
    d = {k: {k} for k in range(n)}
    for i in range(n,2*n-1):
        d[i] = d[z[i-n,0]] | d[z[i-n,1]]
    return d

def get_leaves_dict_lin(lin):
    n = len(lin)
    d = {k: {k} for k in range(n)}
    created = defaultdict(list)
    new_node = n
    levels = {k: 0 for k in range(n)}
    no_infos = defaultdict(set)
    no_infos[0] = {k for k in range(n) if lin[k,0] <= 0}
    for j in range(1,lin.shape[1]):
        counts = defaultdict(list)
        for i in range(n):
            if lin[i,j] > 0:
                counts[lin[i,j]].append(i)
            else:
                no_infos[j] = no_infos[j] | {i}
        for k, v in counts.items():
            if len(v) > 1 and created[v[0]] != v:
                d[new_node] = set(v)
                levels[new_node] = j
                created[v[0]] = v
                new_node += 1

    return d, levels, no_infos


def get_leaves_lca(d, i, j, return_node=False):
    n = int((len(d)+1)/2)
    for k in range(n,2*n-1):
        if {i,j} <= d[k]:
            if return_node:
                return d[k], k
            else:
                return d[k]
    if return_node:
        return set(), -1
    else:
        return set()

def flat_clustering(lineages):
    assert lineages.shape[1] == 9 # assumption: species, genus, ..., superkingdom, no rank, no rank
    fc_result = {}
    rank_names = ['genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
    for rank_j in range(1,lineages.shape[1]-2):
        d = defaultdict(set)
        d['no_info'] = set()
        for sample_i in range(lineages.shape[0]):
            if lineages[sample_i, rank_j] >= 0:
                d[lineages[sample_i, rank_j]] = d[lineages[sample_i, rank_j]] | {sample_i}
            else:
                d['no_info'] = d['no_info'] | {sample_i}
        fc_result[rank_names[rank_j-1]] = d
    return fc_result


def geodesic_distance_poincare_half(x0_mu, x0_sigma, x1_mu, x1_sigma):
    d_mp = torch.sqrt(torch.clamp((x0_mu - x1_mu) ** 2 + (x0_sigma + x1_sigma) ** 2, min=1e-6))
    d_mm = torch.sqrt(torch.clamp((x0_mu - x1_mu) ** 2 + (x0_sigma - x1_sigma) ** 2, min=1e-6))
    d = torch.log(d_mp + d_mm) - torch.log(d_mp - d_mm)
    return torch.sqrt(torch.clamp(torch.sum(d ** 2, -1), min=1e-6))

def multivariate_normal_diag_logpdf(x, mean, logvar): # ok if x, mean, and logvar are broadcastable
    n_bat, n_dim = x.shape
    sse = torch.sum(((x - mean) ** 2)/(1 if logvar == None else torch.exp(torch.clamp(logvar, min=-10, max=10))), 1) # clamp for preventing overflow
    logdet = 0 if logvar == None else torch.sum(logvar,1)
    return -0.5 * n_dim * torch.log(2 * torch.tensor(math.pi)) - 0.5 * logdet - 0.5 * sse

def pseudo_hyperbolic_gaussian_logpdf(x, mu=None, logvar=None): # x, mu in R^(n+1), logvar in R^n ok if they are broadcastable
    mu0 = torch.zeros_like(x)
    mu0[...,0] = 1
    mu = mu0 if mu == None else mu
    v = parallel_transport_inv(exponential_map_inv(x, mu), mu0, mu)[...,1:]
    #print(torch.isnan(v).any())
    n_dim = v.shape[1]
    mean = torch.zeros_like(v)
    r = torch.clamp(geodesic_distance(mu, x), max=10) # clamp for preventing overflow, e.g., sinh(100)=inf, sinh(10)=11013
    #print(torch.squeeze(r))
    return multivariate_normal_diag_logpdf(v, mean, logvar) - (n_dim -1) * (torch.log(torch.sinh(r)) - torch.log(r))

def gaussian_logpdf(x, mu=None, logvar=None):
    mu = torch.zeros_like(x) if mu == None else mu
    return multivariate_normal_diag_logpdf(x, mu, logvar)

def pseudo_gaussian_manifold_normal_format(alpha, log_beta_sq, log_gamma_sq, log_c):
    mu = alpha
    logvar = log_beta_sq + log_gamma_sq
    a = (-(log_gamma_sq + log_c)).exp() / 4 + 1
    logb = - (log_beta_sq + log_gamma_sq + log_c + math.log(4))
    return mu, logvar, a, logb

class DivergenceLoss(torch.nn.Module):
    def __init__(self, latent_space='Euclid', alpha=0, gamma=1, c=1):
        super(DivergenceLoss, self).__init__()
        self.latent_space = latent_space
        #torch.autograd.set_detect_anomaly(True)
        self.alpha = alpha # recommendation: alpha = 1 if encoder and decoder are sufficiently expressive otherwise 0 (refer to InfoVAE)
        self.gamma = gamma # recommendation: gamma = dim(data)/dim(hidden) (refer to InfoVAE)
        self.c = c

    # KL of N(mu1,exp(logvar1)) from N(mu2,exp(logvar2)), divided by dim(mu) (and # of samples in the batch)
    def KLloss_gaussian_diag(self, mu1, logvar1, mu2=None, logvar2=None):
        mu2 = torch.zeros_like(mu1) if mu2 == None else mu2
        logvar2 = torch.zeros_like(logvar1) if logvar2 == None else logvar2
        logvar2 = torch.tensor(logvar2, dtype=logvar1.dtype, device=logvar1.device) if not torch.is_tensor(logvar2) else logvar2
        kl = 0.5 * (logvar2 - logvar1 -mu1.shape[-1] + (logvar1-logvar2).exp() + (mu2 - mu1).pow(2)/logvar2.exp()).mean()
        return kl

    # KL of G(shape1,exp(lograte1)) from G(shape2,exp(lograte2)), divided by dim(shape1)
    def KLloss_gamma_diag(self, shape1, lograte1, shape2, lograte2):
        kl = (shape2 * (lograte1 - lograte2) - (torch.lgamma(shape1) - torch.lgamma(shape2))\
              + (shape1 - shape2) * torch.digamma(shape1) - (1 - (lograte2 - lograte1).exp()) * shape1).mean()
        return kl 

    # KL divergence of the approximate posterior from the prior
    def KLloss(self, x, mu, logvar):
        if self.latent_space == 'Euclid':
            # log_appro_posteorir = gaussian_logpdf(x, mu, logvar)
            # log_model_prior = gaussian_logpdf(x)
            # kl_loss = torch.mean(log_appro_posteorir - log_model_prior)
            kl_loss = self.KLloss_gaussian_diag(mu, logvar)
        elif self.latent_space == 'Lorentz':
            log_appro_posteorir = pseudo_hyperbolic_gaussian_logpdf(x, mu, logvar)
            log_model_prior = pseudo_hyperbolic_gaussian_logpdf(x)
            kl_loss = torch.mean(log_appro_posteorir - log_model_prior)
        elif self.latent_space == 'GaussianManifold' or self.latent_space == 'GaussianManifoldL': 
            alpha, log_beta_sq = mu[...,0], mu[...,1]
            if self.latent_space == 'GaussianManifold':
                log_gamma_sq = logvar
                log_c = math.log(self.c)
                log_c_p = math.log(self.c)
            elif self.latent_space == 'GaussianManifoldL':
                log_gamma_sq, log_c = logvar[...,0], logvar[...,1]
                log_c_p = math.log(1) # another prioir is possible
            normal_mu, normal_logvar, gamma_a, gamma_logb = pseudo_gaussian_manifold_normal_format(alpha, log_beta_sq, log_gamma_sq, log_c)
            alpha_p, log_beta_sq_p, log_gamma_sq_p = torch.zeros_like(alpha), torch.zeros_like(log_beta_sq), torch.zeros_like(log_gamma_sq)
            normal_mu_p, normal_logvar_p, gamma_a_p, gamma_logb_p = pseudo_gaussian_manifold_normal_format(alpha_p, log_beta_sq_p, log_gamma_sq_p, log_c_p)
            kl_loss = self.KLloss_gaussian_diag(normal_mu, normal_logvar, normal_mu_p, normal_logvar_p)\
                      + self.KLloss_gamma_diag(gamma_a, gamma_logb, gamma_a_p, gamma_logb_p)
        return kl_loss

    def compute_kernel(self, x, y):
        return torch.exp(-torch.mean((x[:,None]-y[None,:])**2, -1)/x.shape[-1])

    # MMD distance between the approximate prior and the prior
    def compute_mmd(self, x, y):
        x_kernel  = self.compute_kernel(x, x)
        y_kernel  = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def MMDloss(self, x):
        if self.latent_space == 'Euclid':
            v = x
        if self.latent_space == 'Lorentz':
            mu0 = torch.zeros_like(x)
            mu0[:,0] = 1
            v = exponential_map_inv(x, mu0)
            v = v[:,1:]
        w = torch.randn_like(v)
        return self.compute_mmd(v,w)

    def forward(self, x, mu, logvar):
        div_loss = 0
        if self.alpha != 1:
            div_loss += (1 - self.alpha) * self.KLloss(x, mu, logvar)
        elif self.gamma != 0:
            div_loss += (self.alpha - 1 + self.gamma) * self.MMDloss(x)
            #div_loss += (self.alpha -1 + self.gamma) * (x.shape[-1]**2) * torch.log(self.MMDloss(x))
        return div_loss


class EmbedDistanceLoss(torch.nn.Module):
    def __init__(self, n_samples, n_rep=1, latent_space='Euclid', strategy_emd='slr', strategy_samp='random', strategy_rank='rank', strategy_perp=5.0, strategy_gamm=0.1,
                 strategy_hrms_alph=2, strategy_hrms_beta=2, strategy_hrms_gamm=0.5, min_dist=0.1, lin_trn_counts = None, n_channels=512):
        super(EmbedDistanceLoss, self).__init__()
        self.latent_space = latent_space
        if strategy_emd == 'none':
            self.dist_loss = self.dist_loss_none
        elif strategy_emd == 'direct':
            self.dist_loss = self.dist_loss_direct
        elif strategy_emd == 'slr':
            self.dist_loss = self.dist_loss_slr
            self.n_samples = n_samples
            self.n_rep = n_rep
            self.strategy_samp = strategy_samp
            self.strategy_rank = strategy_rank
        elif 'tsne' in strategy_emd:
            self.dist_loss = self.dist_loss_tsne
            self.get_effective_bits = self.get_effective_bits_tsne
            self.perplexity = strategy_perp
            self.branch = torch.nn.parameter.Parameter(torch.zeros(7)) if strategy_emd == 'tsneL' else None
            self.symmetric = True if strategy_emd == 'tsneS' else False
            self.binomial = True if strategy_emd == 'tsneB' else False
            self.cubic = True if strategy_emd == 'tsneC' else False
            self.repel = True if strategy_emd == 'tsneR' else False
            self.t_dist = True if strategy_emd == 'tsneT' else False
            self.one_deg = True if strategy_emd == 'tsneO' else False
            self.focus = 2 if strategy_emd == 'tsneF' else 1
            if strategy_emd == 'hrmstsne':
                self.eps = 0.1 # 0.1
                self.alph = strategy_hrms_alph # 2 or 2
                self.beta = strategy_hrms_beta # 2 or 50
                self.gamm = strategy_hrms_gamm # 0.5 or 1
                self.dist_loss = self.dist_loss_hrmstsne
        elif strategy_emd == 'umap':
            self.dist_loss = self.dist_loss_umap
            self.get_effective_bits = self.get_effective_bits_umap
            self.perplexity = strategy_perp
            self.min_dist = min_dist
            self.a, self.b = self.find_ab_params(1.0, self.min_dist)
        elif strategy_emd == 'rrl':
            self.dist_loss = self.dist_loss_rrl
        elif strategy_emd == 'cl':
            self.dist_loss = self.dist_loss_cl
            self.margin = 1
            self.min_dist = 2
        elif strategy_emd == 'ce':
            self.dist_loss = self.dist_loss_ce
            lin_trn_unique, counts = lin_trn_counts
            #print(counts)
            u = np.unique(lin_trn_unique[:,1:].reshape(-1))
            u = u[u > 0]
            taxids_dic = {u[i]: i for i in range(len(u))}
            n_classes = len(taxids_dic)
            self.n_classes = n_classes
            self.taxids_multilabel = {}
            for i in range(lin_trn_unique.shape[0]):
                label = np.zeros(n_classes)
                for tid in lin_trn_unique[i,1:]:
                    if tid > 0:
                        label[taxids_dic[tid]] += 1
                self.taxids_multilabel[tuple(lin_trn_unique[i,1:])] = label
            #self.register_buffer('genus_taxids_key', torch.tensor(list(self.genus_taxids_dic.keys()), dtype=torch.int32))
            #weights = torch.tensor(1/counts, dtype=torch.float32)
            #self.loss = torch.nn.CrossEntropyLoss(weights, ignore_index=-1)
            self.loss = torch.nn.BCELoss()
            self.fc = torch.nn.Linear(n_channels * (2 if 'GaussianManifold' in latent_space else 1), n_classes)
        elif strategy_emd == 'dp':
            self.dist_loss = self.dist_loss_dp
            self.n_samples = n_samples
        elif strategy_emd == 'hrms':
            self.dist_loss = self.dist_loss_hrms
            self.eps = 0.1 # 0.1
            self.alph = strategy_hrms_alph # 2 or 2
            self.beta = strategy_hrms_beta # 2 or 50
            self.gamm = strategy_hrms_gamm # 0.5 or 1
        elif 'hrms' in strategy_emd:
            self.dist_loss = self.dist_loss_hrmsE
            self.eps = 0.1 # 0.1
            self.alph = strategy_hrms_alph # 2 or 2
            self.beta = strategy_hrms_beta # 2 or 50
            self.gamm = strategy_hrms_gamm # 0.5 or 1
            self.weighting = True if 'W' in strategy_emd else False
            self.down_p = 0.5 if 'DP' in strategy_emd else 1
            self.down_n = 0.5 if 'DN' in strategy_emd else 1
            self.sne_th = 0.99 if 'S' in strategy_emd else 0
            self.adaptive = True if 'A' in strategy_emd else False
            self.FTM = True if 'FTM' in strategy_emd else False
        elif strategy_emd == 'sneht':
            self.dist_loss = self.dist_loss_sneht
            self.alph = 0.05
        elif strategy_emd == 'sne':
            self.dist_loss = self.dist_loss_sne
        elif 'sneE' in strategy_emd:
            self.dist_loss = self.dist_loss_sneE
        self.gamma = strategy_gamm
        self.distance = partial(embedding_distance, latent_space = latent_space)
        random.seed(0)
        torch.autograd.set_detect_anomaly(True)

    def get_dist_from_comb(self, embed, lineages, idx_comb, c=None):
        idx_comb_l, idx_comb_r = map(list, zip(*idx_comb))
        dist_embed = self.distance(embed[idx_comb_l], embed[idx_comb_r], c=c)
        dist_lin = lineage_distance(lineages[idx_comb_l], lineages[idx_comb_r])
        return dist_embed, dist_lin # all should be 1d tensor

    def dist_loss_none(self, embed, lineages, c=None):
        return torch.tensor(0.0, requires_grad=True).to(embed)

    def dist_loss_direct(self, embed, lineages, c=None):
        idx_comb = list(itertools.combinations(list(range(embed.shape[0])), 2))
        dist_embed, dist_lin = self.get_dist_from_comb(embed, lineages, idx_comb, c=c)
        return torch.mean(torch.abs(dist_embed-dist_lin))

    def dist_loss_slr(self, embed, lineages, c=None):
        idx_comb = list(itertools.combinations(list(range(embed.shape[0])), 2))
        loss = torch.tensor(0, dtype=torch.float32, device=embed.device)
        for rep in range(self.n_rep):
            if self.strategy_samp == 'random':
                idx_comb_sampled = random.sample(idx_comb, self.n_samples)
            elif self.strategy_samp == 'anchor':
                idx_list = list(range(embed.shape[0]))
                anchor_idx = random.sample(idx_list, 1)[0]
                idx_comb_sampled = [(anchor_idx, idx) for idx in random.sample(idx_list[:anchor_idx] + idx_list[anchor_idx+1:], min(self.n_samples, len(idx_list)-1))]

            dist_embed, dist_lin = self.get_dist_from_comb(embed, lineages, idx_comb_sampled, c=c)
            min_pair_idx = torch.argmin(dist_lin)
            if self.strategy_rank == 'dist':
                dist_embed = dist_embed / dist_lin
            elif self.strategy_rank == 'rank':
                dist_embed = dist_embed
            loss += -log_softmax(-dist_embed)[min_pair_idx]
        if not loss.requires_grad:
            loss.requires_grad = True
        return loss/self.n_rep

    # find_ab_params was adapted from https://github.com/lmcinnes/umap/blob/053bdaa47a49b79f44033b801c88d3c743106e1b/umap/umap_.py#L1386
    def find_ab_params(self, spread, min_dist):
        """Fit a, b params for the differentiable curve used in lower
        dimensional fuzzy simplicial complex construction. We want the
        smooth curve (from a pre-defined family with simple gradient) that
        best matches an offset exponential decay.
        """

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    def get_input_similarities_tsne(self, dist_X_neighbors_sq, sigma): # sq: squared
        if self.binomial:
            dist = torch.sqrt(torch.clamp(dist_X_neighbors_sq, min=1e-3))
            p = dist / 14
            entropy = - p * torch.log(torch.clamp(p, min=1e-6)) - (1-p) * torch.log(torch.clamp(1-p, min=1e-6))
            kernel = (2 ** (14 * entropy - 14)) * sigma
            print(torch.mean(kernel))
            return kernel / torch.sum(kernel, dim=1, keepdim=True)
        if self.cubic:
            #return torch.softmax(-(dist_X_neighbors_sq ** (3/2))/(sigma**2), dim=1)
            return torch.softmax(-(torch.clamp(dist_X_neighbors_sq, min=1e-3) ** (1/2)) * math.log(2/3)/2 , dim=1)
        if self.one_deg:
            return torch.softmax(-(torch.clamp(dist_X_neighbors_sq, min=1e-3) ** (0.3/2))/(sigma**2), dim=1)
        if self.t_dist:
            kernel = (1 + (dist_X_neighbors_sq ** (1.5/2))/sigma) ** (-(sigma + 1)/2)
            return kernel / torch.sum(kernel, dim=1, keepdim=True)
        return torch.softmax(-dist_X_neighbors_sq/(sigma**2), dim=1)

    def get_input_similarities_umap(self, dist_X_neighbors_dev, sigma): # dev: deviation
        return torch.exp(-dist_X_neighbors_dev/sigma)

    def get_effective_bits_tsne(self, dist_X_neighbors_sq, sigma):
        prob = self.get_input_similarities_tsne(dist_X_neighbors_sq, sigma)
        effective_bits = -torch.sum(prob * torch.log(torch.clamp(prob, min=1e-6)), dim=1)
        return effective_bits

    def get_effective_bits_umap(self, dist_X_neighbors_dev, sigma):
        effective_bits = torch.sum(self.get_input_similarities_umap(dist_X_neighbors_dev, sigma), dim=1)
        return effective_bits

    def variance_search_from_perplexity(self, dist_X_neighbors, perplexity=1):
        sigma_lr = torch.ones((dist_X_neighbors.shape[0],2), dtype=torch.float32).to(dist_X_neighbors)
        sigma_r_open = torch.ones(dist_X_neighbors.shape[0], device=dist_X_neighbors.device).reshape((-1,1)).to(torch.bool)
        sigma_lr[:,0] = 0.0
        sigma_lr[:,1] = 3.0
        sigma_lr = sigma_lr.to(torch.float32)
        #print(sigma_lr.dtype)
        sigma = torch.ones_like(sigma_lr[:,0]).to(torch.float32).reshape((-1,1))
        n_neighbors_bits = math.log(perplexity, 2)
        for i in range(10):
            effective_bits = self.get_effective_bits(dist_X_neighbors, sigma)
            next_sigma = torch.stack(((sigma_lr[:,0] + sigma[:,0])/2, (sigma_lr[:,1] + sigma[:,0])/2), 1).clone()
            next_idx = (effective_bits < n_neighbors_bits).reshape((-1,1))
            #print(sigma_lr.dtype, sigma.dtype)
            sigma_lr.scatter_(1, (~next_idx).long(), sigma)
            #print(next_sigma.shape)
            #print(next_idx.shape)
            #print(sigma.shape)
            sigma = torch.gather(next_sigma, 1, next_idx.long())
            #print(sigma)
            r_open = (sigma_r_open & next_idx).reshape(-1)
            #print(r_open)
            #print(sigma.shape)
            #print(sigma_lr.dtype)
            #print(sigma_lr)
            #print(sigma.dtype)
            #print(sigma.shape)
            sigma_lr[r_open,1] = 3.0 * sigma.reshape(-1)[r_open]
            sigma_r_open = sigma_r_open & next_idx
        return sigma

    def get_off_diagonal_elements(self, M):
        return M[~torch.eye(*M.shape, dtype = torch.bool)].reshape((M.shape[0], -1))

    def fill_off_diagonal_elements(self, M):
        N = torch.zeros((M.shape[0], M.shape[0]), device=M.device)
        N[~torch.eye(*N.shape, dtype=torch.bool)] = M.reshape(-1)
        return N

    def get_embed_similarities_tsne(self, dist_embed, a=1, b=1):
        w = self.gamma * (self.gamma ** 2 + a * dist_embed ** (2*b)) ** (-1)
        return w / torch.sum(w)

    def get_embed_similarities_umap(self, dist_embed, a=1, b=1):
        w = self.gamma * (self.gamma ** 2 + a * dist_embed ** (2*b)) ** (-1)
        return w

    def dist_loss_tsne(self, embed, lineages, c=None):
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        if self.branch is not None:
            new_branch = torch.zeros_like(dist_input)
            for i in range(7):
                new_branch = (dist_input >= i * 2).long() * (2 * torch.exp(self.branch[i]))
            dist_input = new_branch
        # get similarities p and q
        dist_input_sq = dist_input ** 2
        sigma = self.variance_search_from_perplexity(dist_input_sq, self.perplexity)
        prob = self.get_input_similarities_tsne(dist_input_sq, sigma)
        prob = self.fill_off_diagonal_elements(prob)
        prob = (prob + prob.T)/ (2*prob.shape[0])
        p = self.get_off_diagonal_elements(prob)
        q = self.get_embed_similarities_tsne(dist_embed)
        loss = - torch.sum(p * torch.log(q) * (1 if self.focus == 1 else (1-q)**self.focus))
        if self.branch is not None:
            loss += torch.sum(p * torch.log(torch.clamp(p, min=1e-6)))
        if self.symmetric:
            loss += torch.sum(q * (torch.log(q)-torch.log(torch.clamp(p, min=1e-6))))
        if self.repel:
            n_pairs = p.shape[0] * p.shape[1]
            loss += - torch.sum((1-p)/(n_pairs-1) * torch.log((1-q)/(n_pairs-1)))
        return loss

    # see Appendix C in "Leland McInnes and John Healy and James Melville. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, 2020."
    def dist_loss_umap(self, embed, lineages, c=None):
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        # get similarities v and w
        dist_input_dev = dist_input - torch.min(dist_input, dim=1, keepdim=True)[0]
        sigma = self.variance_search_from_perplexity(dist_input_dev, self.perplexity)
        prob = self.get_input_similarities_umap(dist_input_dev, sigma)
        prob = self.fill_off_diagonal_elements(prob)
        prob = prob + prob.T - prob * prob.T
        v = self.get_off_diagonal_elements(prob)
        w = self.get_embed_similarities_umap(dist_embed, a=self.a, b=self.b)
        return -torch.sum(v * torch.log(w)) #+ (1-v) * torch.log(1-w))

    def ranked_list_loss(self, dist_embed, p_margin, n_margin, p_mask, n_mask):
        return torch.clamp((dist_embed-p_margin) * p_mask, min=0.0) + torch.clamp((n_margin-dist_embed) * n_mask, min=0.0)

    def dist_loss_rrl(self, embed, lineages, c=None):
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        loss = 0
        p_margin = torch.exp(dist_input/10) - 0.5
        n_margin = torch.exp(dist_input/10)
        for j in range(1, lineages.shape[1]):
            mask = (dist_input <= 2*j).long()
            loss += self.ranked_list_loss(dist_embed, p_margin, n_margin, mask, 1-mask)
        #mask = torch.zeros_like(dist_embed)
        #print(mask.shape)
        #mask.scatter_(1, torch.argmin(dist_input, dim=1, keepdim=True), 1)
        #rrl = self.ranked_list_loss(dist_embed, dist_input, dist_input, mask, 1-mask)
        if isinstance(loss, int):
            loss = self.dist_loss_none(embed, lineages)
        return torch.mean(loss)

    def multiclass_ht(self, p, target, ignore_index=-1, correction='Bonferroni'):
        u, inv = np.unique(target, return_inverse=True)
        n_class = len(u)
        class_mask = [(inv == i) for i in range(n_class)]
        p_merged = torch.cat([torch.sum(p[:,class_mask[i]], 1, keepdim=True) for i in range(n_class)], 1)
        p_cm = torch.cat([torch.mean(p_merged[class_mask[i], :], 0, keepdim=True) for i in range(n_class)], 0)
        p_diag = torch.zeros(n_class) # here p stands for positive
        p_diag[u == ignore_index] = 0
        p_mask = torch.diag(p_diag).to(p.device)
        n_mask = 1 - torch.eye(n_class).to(p.device)
        if correction == 'Bonferroni':
            return self.ranked_list_loss(1-p_cm, self.alph, 1-self.alph/n_class, p_mask, n_mask)
        
    def get_embed_similarities_sneht(self, dist_embed, a=1, b=1):
        w = self.gamma * (self.gamma ** 2 + a * dist_embed ** (2*b)) ** (-1)
        return w / torch.sum(w, 1, keepdim=True)

    def dist_loss_sneht(self, embed, lineages, c=None):
        lin = lineages.detach().cpu().numpy()
        dist_embed = self.distance(embed[:,None], embed[None], c=c)
        q = self.get_embed_similarities_sneht(self.distance(embed[:,None], embed[None], c=c))
        # q.fill_diagonal_(0)
        loss = 0
        for j in range(1,7): # genus, family, order, class, phylum, superkingdom
            loss += torch.mean(self.multiclass_ht(q, lin[:,j]))
        if isinstance(loss, int):
            loss = self.dist_loss_none(embed, lineages)
        return loss

    def sne_kernel(self, inp, kernel='exp', a=1, b=1):
        if kernel=='exp':
            return torch.softmax(torch.exp(-inp**2), dim=1)
        elif kernel=='student':
            w = self.gamma * (self.gamma ** 2 + a * inp ** (2*b)) ** (-1)
            return w / torch.sum(w, 1, keepdim=True)

    def dist_loss_sne(self, embed, lineages, c=None):
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        prob = self.sne_kernel(dist_embed)
        loss = 0
        for j in range(1,7):
            p_mask = (dist_input <= 2 * j).long()
            n_neighbor = torch.sum(p_mask, 1, keepdim=True)
            weights = torch.ones_like(n_neighbor) / torch.clamp(n_neighbor, min=1) # for macro average
            loss += -torch.sum(p_mask * prob / weights) / torch.sum(weights) # each level have the same weights
        if isinstance(loss, int):
            loss = self.dist_loss_none(embed, lineages)
        return loss

    def dist_loss_sneE(self, embed, lineages, c=None):
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        # sne kernel
        a = b = 1
        inp = dist_embed
        w = self.gamma * (self.gamma ** 2 + a * inp ** (2*b)) ** (-1)
        prob = w
        for rank_j in range(6,0,-1): # genus, family, order, class, phylum, superkingdom
            eps = self.eps * rank_j
            alph = self.alph / rank_j
            beta = self.beta / rank_j
            # HRMS pair miner
            p_mask = dist_input <= rank_j * 2
            n_mask = ~p_mask
            p_hard = (p_mask & (torch.min(n_mask.long() * dist_embed, 1, keepdim=True)[0] - eps < dist_embed)).long()
            n_hard = (n_mask & (torch.max(p_mask.long() * dist_embed, 1, keepdim=True)[0] + eps > dist_embed)).long()
            n_pair = torch.sum(p_hard) + torch.sum(n_hard)
            # HRMS loss function
            if self.adaptive:
                p_gamma = (torch.min(n_mask.long() * dist_embed, 1, keepdim=True)[0] - eps).detach()
                n_gamma = torch.max(p_mask.long() * dist_embed, 1, keepdim=True)[0] + eps
            else:
                p_gamma = 0.1
                n_gamma = 0.1
            dist_embed_div_p = dist_embed - p_gamma
            dist_embed_div_n = dist_embed - n_gamma


    def dist_loss_cl(self, embed, lineages, c=None):
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        mask = torch.zeros_like(dist_embed)
        #print(mask.shape)
        #mask.scatter_(1, torch.argmin(dist_input, dim=1, keepdim=True), 1)
        mask = (dist_input <= self.min_dist).to(torch.long)
        cl = self.ranked_list_loss(dist_embed, 0, self.margin, mask, 1-mask)
        return torch.sum(cl)


    def dist_loss_ce(self, embed, lineages, c=None):
        #lin = lineages[:,1].detach()
        #mask = torch.isin(lin, self.genus_taxids_key)
        #lin[~mask] = -1
        #lin = lineages[:,1].cpu().detach().numpy()
        lin = lineages[:,1:].cpu().detach().numpy()
        #encoding_index = torch.tensor([self.genus_taxids_dic.get(taxid, -1) for taxid in lin], 
        #                              dtype=torch.long, device=embed.device)
        #print(encoding_index)l
        encoding_index = torch.tensor([self.taxids_multilabel.get(tuple(l), [0]*self.n_classes) for l in lin], 
                                       dtype=torch.float32, device=embed.device)
        loss = self.loss(torch.sigmoid(self.fc(embed.reshape((embed.shape[0], -1)))), encoding_index)
        #print(loss)
        if torch.isnan(loss):
            return self.dist_loss_none(embed, lineages)
        return loss

    def dist_loss_dp(self, embed, lineages, c=None):
        # construct hierarchy from embeddings
        dist_embeddings_torch = self.distance(embed[:,None], embed[None], c=c)
        dist_embeddings = dist_embeddings_torch.cpu().detach().numpy()
        r,c = np.triu_indices(dist_embeddings.shape[0],1)
        pdist_embeddings = dist_embeddings[r,c]
        Z = linkage(pdist_embeddings, 'ward')
        d = get_leaves_dict(Z[:,0:2].astype(int))

        # flat clustering of the true tree at each rank
        #fc_results = flat_clustering(lineages)
        #loss = self.dist_loss_none(embed, lineages)
        #margin = dist_embeddings_torch.detach()
        #bs = embed.shape[0]
        #key = random.choice(list(fc_results.keys())) # this is an experimental setting
        #for rank_name, fc_results in fc_result.items():
        #    if rank_name != key:
        #        continue
        #    n_pairs = 0
        #    no_infos = fc_results.pop('no_info')
        #    #p_mask = torch.zeros_like(dist_embed)
        #    #n_mask = torch.zeros_like(dist_embed)
        #    for taxid, s_true in flat_clust_dict.items():
        #        for i,j in itertools.combinations(s_true, 2):
        #            # note: the result of get_leaves_lca may contain species which have no information in the rank and thus are excluded from s_pred
        #            s_pred = get_leaves_lca(d, i, j) - no_infos
        #            p_set = s_true - s_pred
        #            n_set = s_pred - s_true
        #            p_mask = torch.zeros(bs, device=embed.device)
        #            p_mask[list(p_set)] = 1
        #            n_mask = torch.zeros(bs, device=embed.device)
        #            n_mask[list(n_set)] = 1
        #            #p_mask[np.ix_([i,j], list(p_set))] += 1 # voting in the same cluster (maximum: |s_true|-1)
        #            #n_mask[np.ix_([i,j], list(n_set))] += 1
        #            loss += self.ranked_list_loss(dist_embeddings_torch[[i,j]], margin[i,j], margin[i,j], p_mask, n_mask)
        #        n_pairs += len(s_true) * (len(s_true) - 1) / 2
        #        # we do not treat each group with equal importance; rather, treat each suggestion as equal
        #    loss /= n_pairs
        
        bs = embed.shape[0]
        margin = dist_embeddings_torch.detach()
        d_lin, levels, no_infos = get_leaves_dict_lin(lineages.cpu().detach().numpy())
        idx_comb = list(itertools.combinations(list(range(bs)), 2))
        idx_comb_sampled = idx_comb #random.sample(idx_comb, self.n_samples)
        loss = 0
        for i,j in idx_comb_sampled:
            s_true, node  = get_leaves_lca(d_lin, i, j, return_node=True)
            if len(s_true) == 0: # in case i and j are disconnected in the lineage
                continue
            no_info = no_infos[levels[node]]
            # note: the result of get_leaves_lca may contain species which have no information in the rank and thus are excluded from s_pred
            s_pred = get_leaves_lca(d, i, j) #- no_info
            p_set = s_true - s_pred - no_info
            n_set = s_pred - s_true
            p_mask = torch.zeros(bs, device=embed.device)
            p_mask[list(p_set)] = 1
            n_mask = torch.zeros(bs, device=embed.device)
            n_mask[list(n_set)] = 1
            #print(i,j)
            #print(p_set)
            #print(n_set)
            loss += torch.sum(self.ranked_list_loss(dist_embeddings_torch[[i,j],:], margin[i,j], margin[i,j], p_mask, n_mask))

        if isinstance(loss, int):
            loss = self.dist_loss_none(embed, lineages)

        return loss/self.n_samples

    def dist_loss_hrms(self, embed, lineages, c=None):
        loss = 0
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        gamm = self.gamm
        for rank_j in range(1,7): # genus, family, order, class, phylum, superkingdom
            eps = self.eps * rank_j
            alph = self.alph / rank_j
            beta = self.beta / rank_j
            # HRMS pair miner
            p_mask = dist_input <= rank_j * 2
            n_mask = ~p_mask
            p_hard = (p_mask & (torch.min(n_mask.long() * dist_embed, 1, keepdim=True)[0] - eps < dist_embed)).long()
            n_hard = (n_mask & (torch.max(p_mask.long() * dist_embed, 1, keepdim=True)[0] + eps > dist_embed)).long()
            n_pair = torch.sum(p_hard) + torch.sum(n_hard)
            # HRMS loss function
            dist_embed_div = dist_embed - gamm
            loss += torch.sum(torch.log(torch.sum(torch.exp( alph   * dist_embed_div * p_hard) * p_hard, 1) + 1) / alph
                            + torch.log(torch.sum(torch.exp((-beta) * dist_embed_div * n_hard) * n_hard, 1) + 1) / beta) / n_pair

        if isinstance(loss, int):
            loss = self.dist_loss_none(embed, lineages)

        return loss

    def dist_loss_hrmsE(self, embed, lineages, c=None):
        loss = 0
        dist_embed = self.get_off_diagonal_elements(self.distance(embed[:,None], embed[None], c=c))
        dist_input = self.get_off_diagonal_elements(lineage_distance(lineages[:,None], lineages[None]))
        gamm = self.gamm
        if self.sne_th > 0 or self.FTM:
            a = b = 1
            inp = dist_embed
            w = self.gamma * (self.gamma ** 2 + a * inp ** (2*b)) ** (-1)
            prob = w
            #prob = self.sne_kernel(dist_embed, kernel='student')
            #print(torch.sum(prob, 1))
        confidence = torch.ones(1, device=dist_embed.device)
        if self.FTM:
            score = torch.zeros((embed.shape[0], 6), device=embed.device)
            weights_mask = torch.zeros_like(score)
            score_coherent = torch.zeros_like(score)
            self.focus = 2
        for rank_j in range(6,0,-1): # genus, family, order, class, phylum, superkingdom
            eps = self.eps * rank_j
            alph = self.alph / rank_j
            beta = self.beta / rank_j
            # HRMS pair miner
            p_mask = dist_input <= rank_j * 2
            n_mask = ~p_mask
            p_hard = (p_mask & (torch.min(n_mask.long() * dist_embed, 1, keepdim=True)[0] - eps < dist_embed)).long()
            n_hard = (n_mask & (torch.max(p_mask.long() * dist_embed, 1, keepdim=True)[0] + eps > dist_embed)).long()
            if self.down_p < 1:
                p_hard = p_hard * torch.bernoulli(torch.ones_like(p_hard) * (1 - (1-self.down_p) * (rank_j-1)/5))
            if self.down_n < 1:
                n_hard = n_hard * torch.bernoulli(torch.ones_like(n_hard) * (self.down_n + (1-self.down_n) * (rank_j-1)/5))
            n_pair = torch.sum(p_hard) + torch.sum(n_hard)
            # HRMS loss function
            if self.adaptive:
                p_gamma = (torch.min(n_mask.long() * dist_embed, 1, keepdim=True)[0] - eps).detach()
                n_gamma = (torch.max(p_mask.long() * dist_embed, 1, keepdim=True)[0] + eps).detach()
            else:
                p_gamma = gamm
                n_gamma = gamm
            dist_embed_div_p = dist_embed - p_gamma
            dist_embed_div_n = dist_embed - n_gamma
            
            if self.sne_th > 0 or self.FTM:
                p_true = p_mask.long()
                p_pred = (torch.max(p_true * dist_embed, 1, keepdim=True)[0] + eps >=  dist_embed).long()
                prob_sum = torch.sum(prob * p_pred, 1)
                prob_sum1 = prob_sum.clone()
                prob_sum1[torch.sum(p_pred, 1) == 0] = 1
                n_neighbor = torch.sum(p_true, 1, keepdim=True)
                #print('n_neighbor: ', n_neighbor.reshape(-1))
                weights = 1 / (n_neighbor+1) # for macro average
                #print('weights: ', weights.reshape(-1))
                ma_prec = torch.sum(p_true * prob, 1) / prob_sum1
                ma_prec1 = ma_prec.clone()
                #print('maprec1: ', ma_prec)
                ma_prec1[n_neighbor.reshape(-1) == 0] = 1
                #print('maprec2: ', ma_prec)

                if self.sne_th > 0:
                    ma_prec2 = torch.sum(ma_prec1.reshape(-1,1) * weights) / torch.sum(weights)
                    print(ma_prec2.item())
                    #confidence *= ma_prec.item()
                    loss += - torch.log(torch.clamp(confidence.clone() * ma_prec2, min=1e-3))
                    confidence *= ma_prec2

                if self.FTM:
                    p_mask_mat = self.fill_off_diagonal_elements(p_true.to(torch.float32))
                    p_mask_mat.fill_diagonal_(1)
                    score[:,rank_j-1:rank_j] = torch.matmul(p_mask_mat, ma_prec1.reshape(-1,1) * weights)
                    weights_mask[:,rank_j-1:rank_j] = weights
                    if rank_j == 6:
                        score_coherent[:,rank_j-1:rank_j] = score[:,rank_j-1:rank_j]
                    else:
                        score_coherent[:,rank_j-1:rank_j] = torch.minimum(score[:,rank_j:rank_j+1].clone(), score_coherent[:,rank_j:rank_j+1].clone())
          
            loss += torch.sum(torch.log(torch.sum(torch.exp( alph   * dist_embed_div_p * p_hard) * p_hard, 1) + 1) / alph
                            + torch.log(torch.sum(torch.exp((-beta) * dist_embed_div_n * n_hard) * n_hard, 1) + 1) / beta) / n_pair * (math.exp(rank_j) if self.weighting else 1)

            #if self.sne_th > 0 and ma_prec.item() < self.sne_th:
            #    print('maprec: ', ma_prec.item())

        if self.FTM:
            loss += - torch.sum(torch.log(torch.clamp(score_coherent, min=1e-3)) * ((1-score_coherent)**self.focus) * weights_mask)

        if isinstance(loss, int):
            loss = self.dist_loss_none(embed, lineages)

        return loss
    
    def dist_loss_hrmstsne(self, embed, lineages, c=None):
        return self.dist_loss_hrms(embed, lineages, c=c) + self.dist_loss_tsne(embed, lineages, c=c)

    def forward(self, embed, lineages, c=None):
        return self.dist_loss(embed, lineages, c=c)

def Lorentz_inner_product(x1, x2): # this works for broadcastable x1 and x2 with 1 or more dim
    return torch.sum(x1[...,1:] * x2[...,1:], dim=-1, keepdim=True) - x1[...,0:1] * x2[...,0:1]

def Lorentz_norm(x):
    #print(torch.isnan(Lorentz_inner_product(x,x)).any())
    #print(torch.squeeze(Lorentz_inner_product(x,x)))
    return torch.sqrt(torch.clamp(Lorentz_inner_product(x,x), min=1e-6)) # clamp for gradient calculation

def exponential_map(x, mu): # condition: <x, mu> = 0 and <mu,mu> is reasonably small
    #print('em')
    #print(torch.isnan(x).any())
    #print(torch.isnan(mu).any())
    x_normed = torch.clamp(Lorentz_norm(x), max=10) # clamp for preventing overflow, e.g., cosh(100) = inf, cosh(10) = 11013
    #print(x_normed.shape)
    #x_normed_scaled = torch.sigmoid(x_normed)
    #print(torch.squeeze(x_normed)) 
    #print(torch.isnan(x_normed).any())
    #print(torch.isnan(torch.cosh(x_normed)).any())
    #print(torch.isnan(torch.sinh(x_normed)).any())
    #z = torch.cosh(x_normed) * mu + torch.sinh(x_normed) * x / x_normed
    #z[...,0] = torch.sqrt(1 + torch.sum(z[...,1:]**2, dim=-1))
    #print(torch.isinf(x).any())
    #print(torch.isinf(mu).any())
    #print(torch.isinf(torch.cosh(x_normed) * mu + torch.sinh(x_normed) * x / x_normed).any())
    return torch.cosh(x_normed) * mu + torch.sinh(x_normed) * x / x_normed

def acosh_clamped(x):
    return torch.acosh(torch.clamp(x, min=1.0 + 1e-6)) # clamp for domain of acosh

def exponential_map_inv(x, mu):
    alpha = torch.clamp(- Lorentz_inner_product(mu, x), min=1.0+1e-6) # alpha = cosh(d(x,mu)) = cosh(||u||) >= 1
    #print('em_inv')
    #print(torch.isnan(x).any())
    #print(torch.isnan(mu).any())
    #print(torch.max(x, dim=-1))
    #print(torch.max(mu, dim=-1))
    #print(torch.min(x, dim=-1))
    #print(torch.min(mu, dim=-1))
    #print(torch.isnan(- Lorentz_inner_product(mu, x)).any())
    #print(torch.isnan(alpha).any())
    #print(torch.isnan(acosh_clamped(alpha)/torch.sqrt(alpha**2 -1) * (x - alpha * mu)).any())
    return acosh_clamped(alpha)/torch.sqrt(alpha**2 -1) * (x - alpha * mu)

def parallel_transport(x, mu1, mu2):
    #print(torch.squeeze(Lorentz_inner_product(mu1, mu1)))
    #print(torch.squeeze(Lorentz_inner_product(mu2, mu2)))
    #print('pt')
    alpha = - Lorentz_inner_product(mu1, mu2)
    #print(torch.squeeze(alpha+1))
    #print(torch.isinf(x).any())
    #print(torch.isinf(mu1).any())
    #print(torch.isinf(mu2).any())
    #print(torch.isinf(x + Lorentz_inner_product(mu2 - alpha * mu1, x) * (mu1 + mu2) / (alpha + 1)).any())
    return x + Lorentz_inner_product(mu2 - alpha * mu1, x) * (mu1 + mu2) / (alpha + 1)

def parallel_transport_inv(x, mu1, mu2):
    #print('pt_inv')
    #print(torch.isnan(x).any())
    #print(torch.isnan(mu1).any())
    #print(torch.isnan(mu2).any())
    #print(torch.isnan(parallel_transport(x, mu2, mu1)).any())
    return parallel_transport(x, mu2, mu1)

def geodesic_distance(x1, x2):
    return torch.squeeze(acosh_clamped(- Lorentz_inner_product(x1, x2)), dim=-1)

def geodesic_distance_poincare(x1, x2):
    x1[...,1:] = x1[...,1:] / torch.clamp(x1[...,0:1]+1, min=1e-6)
    x2[...,1:] = x2[...,1:] / torch.clamp(x2[...,0:1]+1, min=1e-6)
    temp = torch.sum((x1[...,1:]-x2[...,1:])**2, -1)/torch.clamp((1-torch.sum(x1[...,1:]**2, -1)) * (1-torch.sum(x2[...,1:]**2, -1)), min=1e-6)
    return torch.squeeze(acosh_clamped(1+2*temp), dim=-1)

def gaussian_sample(latent_space, mu, log_var, n_samples=None):

    #print(torch.isnan(mu).any())
    #print(torch.isnan(log_var).any())

    z = mu
    std = torch.exp(0.5*log_var)
    if n_samples == None:
        eps = torch.randn_like(std)
    else:
        eps = torch.randn((n_samples, std.shape[-1]), device=std.device)
    eps_scaled = eps * std
    if latent_space == 'Euclid':
        z = mu + eps_scaled # parallel transport on Euclid space
    elif latent_space == 'Lorentz':
        v = torch.nn.functional.pad(eps_scaled, (1,0))
        #print(torch.isnan(eps_scaled).any())
        #print(torch.isnan(v).any())
        mu0 = torch.zeros_like(v)
        mu0[...,0] = 1
        #print(v.shape)
        #print(mu0.shape)
        #print(mu.shape)
        #print(torch.isnan(parallel_transport(v, mu0, mu)).any())
        #print(torch.squeeze(Lorentz_inner_product(v, mu0)))
        #print(torch.squeeze(Lorentz_inner_product(parallel_transport(v, mu0, mu), mu)))
        z = exponential_map(parallel_transport(v, mu0, mu), mu)

    #print(z.shape)
    #print(torch.isnan(z).any())

    return z

def log_gamma_sample(a, logb):
    return torch._standard_gamma(a).log() - logb

# define the basic module

#def Normalize(in_channels, num_groups=32):
#    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class cnn_module_encoder(nn.Module):
    def __init__(self, kmer=6, dr=0.2, latent_channels=512, latent_space='Euclid', n_chunk=1, c=1.0):
        super(cnn_module_encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.c = c
        self.n_chunk = n_chunk

        #self.dropout = nn.Dropout(dr)

        # end
        z_channels = 1 # max(1, 10-kmer)
        #self.norm_out = Normalize(256)
        self.conv_out = torch.nn.Conv2d(256, z_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(z_channels)

        self.fc = nn.Linear((2**kmer // 8) * (2**kmer // 8) * z_channels, latent_channels * n_chunk)
        #torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=0.01)

        self.latent_space = latent_space

    def forward(self, x): # (B, 1, 2^kmer, 2^kmer)
        #print(x.shape)
        x = self.bn1(self.relu(self.conv1(x)))
        #print(x.shape)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.bn2(self.relu(self.conv2(x)))
        #print(x.shape)
        x = self.bn3(self.conv_out(x))
        #print(x.shape)

        x = torch.flatten(x, 1)
        embeds = torch.chunk(self.fc(x), self.n_chunk, dim=-1)
        
        # apply the space-specific final layer and sample from distribution with embedded parameters
        if self.latent_space == 'Euclid':
            mu, logvar = embeds
            x = gaussian_sample(self.latent_space, mu, logvar) if self.training else mu
        elif self.latent_space == 'Lorentz': # map mu to hyperbolic space
            mu, logvar = embeds
            mu = 3.0 * torch.tanh(mu)
            log_var = 10.0 * torch.tanh(logvar)
            #mu_normed = torch.linalg.vector_norm(mu, dim=-1, keepdim=True)
            #mu_temp = mu
            #mu = torch.cat([torch.cosh(mu_normed), torch.sinh(mu_normed) * mu / mu_normed], dim=-1)
            #print(torch.squeeze(Lorentz_inner_product(mu, mu)))
            v = torch.nn.functional.pad(mu, (1,0))
            mu0 = torch.zeros_like(v)
            mu0[...,0] = 1
            mu = exponential_map(v, mu0)
            #print(torch.squeeze(Lorentz_inner_product(mu, mu)))
            embeds = (mu, logvar)
            x = gaussian_sample(self.latent_space, mu, logvar) if self.training else mu
        elif self.latent_space == 'GaussianManifold' or self.latent_splace == 'GaussianManifoldL':
            if self.latent_space == 'GaussianManifold':
                alpha, log_beta_sq, log_gamma_sq = embeds
                log_c = math.log(self.c)
                logvar = log_gamma_sq
            elif self.latent_space == 'GaussianManifoldL':
                alpha, log_beta_sq, log_gamma_sq, log_c = embeds
                logvar = torch.stack([log_gamma_sq, log_c], dim=-1)
            mu = torch.stack([alpha, log_beta_sq], dim=-1)
            normal_mu, normal_logvar, gamma_a, gamma_logb = pseudo_gaussian_manifold_normal_format(alpha, log_beta_sq, log_gamma_sq, log_c)
            x_mu = gaussian_sample('Euclid', normal_mu, normal_logvar)
            x_logvar = log_gamma_sample(gamma_a, gamma_logb)
            x = torch.stack([x_mu, x_logvar], dim=-1)

        return x, mu, logvar

class cnn_module_decoder(nn.Module):
    def __init__(self, kmer=6, dr=0.1, latent_channels=512, latent_space='Euclid', n_chunk=1):
        super(cnn_module_decoder, self).__init__()
        self.z_channels = 1 # max(1, 10-kmer)
        self.image_size = 2**kmer // 8
        self.fc = nn.Linear(latent_channels * n_chunk, (2**kmer // 8) * (2**kmer // 8) * self.z_channels)
        self.bn_fc = nn.BatchNorm2d(self.z_channels)
        self.conv_in = torch.nn.Conv2d(self.z_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(256)
        self.conv1 = nn.ConvTranspose2d(256,256,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.upsample = partial(torch.nn.functional.interpolate, scale_factor=2.0, mode="nearest")
        self.conv2 = nn.ConvTranspose2d(256,128,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv_out = nn.Conv2d(128,1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        #self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dr)

        torch.nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        self.latent_space = latent_space

    def forward(self, x):
        if self.latent_space == 'Lorentz':
            mu0 = torch.zeros_like(x)
            mu0[...,0] = 1
            x = exponential_map_inv(x, mu0)[...,1:]
        elif self.latent_space == 'GaussianManifold' or self.latent_space == 'GaussianManifoldL':
            x = torch.cat((x[...,0], x[...,1]), -1)
        x = self.bn_fc(self.fc(x).view(-1, self.z_channels, self.image_size, self.image_size))
        x = self.bn_in(self.conv_in(x))
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.upsample(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.conv_out(x)
        #x = torch.sigmoid(x)
        return x

class CNN(nn.Module):
    def __init__(self, kmer=6, dr=0.2, latent_channels=512, latent_space='Euclid'):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        #self.dropout = nn.Dropout(dr)
        
        # end
        z_channels = 1 # max(1, 10-kmer)
        #self.norm_out = Normalize(256)
        self.conv_out = torch.nn.Conv2d(256, z_channels, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear((2**kmer // 8) * (2**kmer // 8) * z_channels, latent_channels * (2 if 'GaussianManifold' in latent_space else 1))
        #torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=0.01)

        self.latent_space = latent_space

    def forward(self, x): # (B, 1, 2^kmer, 2^kmer)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv_out(x)
        #x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.latent_space == 'Lorentz': # map mu to hyperbolic space
            x = 3.0 * torch.tanh(x)
            v = torch.nn.functional.pad(x, (1,0))
            mu0 = torch.zeros_like(v)
            mu0[...,0] = 1
            x = exponential_map(v, mu0)
            #print(torch.squeeze(Lorentz_inner_product(mu, mu)))
        elif 'GaussianManifold' in self.latent_space:
            x = x.reshape(x.shape[0], -1, 2)

        return x

class VAE(nn.Module):
    def __init__(self, kmer=6, dr=0, latent_channels=512, latent_space='Euclid', c = 1):
        super(VAE, self).__init__()
        n_chunk_enc, n_chunk_dec = 1, 1
        if latent_space == 'Euclid' or latent_space == 'Lorentz':
            n_chunk_enc = 2 # mu and logvar
            n_chunk_dec = 1
        elif latent_space == 'GaussianManifold':
            n_chunk_enc = 3 # alpha, log_beta_sq, and log_gamma_sq
            n_chunk_dec = 2
        elif latent_space == 'GaussianManifoldL':
            n_chunk_enc = 4 # alpha, log_beta_sq, log_gamma_sq, (learnable) log_curvature
            n_chunk_dec = 2
        self.encoder = cnn_module_encoder(kmer, dr, latent_channels, latent_space, n_chunk_enc, c)
        self.decoder = cnn_module_decoder(kmer, dr, latent_channels, latent_space, n_chunk_dec)

    def forward(self, x):
        embed_x, embed_mu, embed_logvar = self.encoder(x)
        reconstruction = self.decoder(embed_x)
        return reconstruction, embed_x, embed_mu, embed_logvar

# class VQ is adapted from https://www.kaggle.com/code/maunish/training-vq-vae-on-imagenet-pytorch
class VQ(nn.Module):
    
    def __init__(self, lineages_trn_unique, embedding_dim=64, commitment_cost=0.25, latent_space='Euclid'):
        super().__init__()

        #self.genus_taxids_dic = {self.lineages_trn_unique[i,1]: i for i in range(self.num_embeddings)}
        lineages = torch.from_numpy(lineages_trn_unique).clone().to(torch.int32)
        lineages[:,0] = -1 # make sure lineage is anonymous at species level
        self.register_buffer('lineages', lineages)
        self.distance=partial(embedding_distance, latent_space = latent_space)
        self.num_embeddings = self.lineages.shape[0]
        #self.genus_taxids_dic = {self.lineages[i,1]: i for i in range(self.num_embeddings)}
        self.genus_taxids_dic = {lineages_trn_unique[i,1]: i for i in range(self.num_embeddings)}
        self.register_buffer('genus_taxids_key', torch.tensor(list(self.genus_taxids_dic.keys()), dtype=torch.int32))
        #print(self.genus_taxids_dic.keys())
        self.embedding_dim = embedding_dim * (2 if 'GaussianManifold' in latent_space else 1)
        self.commitment_cost = commitment_cost
        self.latent_space = latent_space

        self.embeddings = nn.Embedding(self.num_embeddings,self.embedding_dim)
        # initialization
        if latent_space == 'Euclid' or 'GaussianManifold':
            torch.nn.init.zeros_(self.embeddings.weight)
        elif latent_space == 'Lorentz':
            torch.nn.init.zeros_(self.embeddings.weight)
            self.embeddings.weight[:,0] = 1
        #self.embeddings.weight.data.uniform_(-1/self.num_embeddings,1/self.num_embeddings)

    def get_codebook_infos(self):
        x = self.convert_codebook(self.embeddings.weight)
        lin = self.lineages
        #print(lin.dtype)
        return x, lin

    def convert_codebook(self, x):
        if self.latent_space == 'Lorentz': # map mu to hyperbolic space
            x = 3.0 * torch.tanh(x)
            v = torch.nn.functional.pad(x, (1,0))
            mu0 = torch.zeros_like(v)
            mu0[...,0] = 1
            x = exponential_map(v, mu0)
            #print(torch.squeeze(Lorentz_inner_product(mu, mu)))
        elif 'GaussianManifold' in self.latent_space:
            x = x.reshape(x.shape[0], -1, 2)
        return x
    
    def forward(self, inputs_all, lineages_all):
        #print(torch.mean(inputs_all))
        #print(torch.mean(lineages_all))
        device = inputs_all.device
        mask = torch.isin(lineages_all[:,1], self.genus_taxids_key)
        #print(lineages_all[:10,1])
        #print(mask.long()[:10])
        inputs = inputs_all[mask]
        inputs_invalid = inputs_all[~mask] # invalid inputs behave like codebook by themselves
        input_shape = inputs.shape
        if input_shape[0] > 0:
            #print('ok0')
            lineages = lineages_all[mask, 1].cpu().detach().numpy()
        
            encoding_index = torch.tensor([self.genus_taxids_dic[taxid] for taxid in lineages], dtype=torch.int32, device=device) 
            #print('ok1')
            quantized = torch.index_select(self.embeddings.weight,0,encoding_index)
            #print('ok2')
            quantized = self.convert_codebook(quantized)
       
            #print(torch.mean(quantized))
            #print(torch.mean(inputs))

            e_latent_loss = self.distance(quantized.detach(),inputs)
            q_latent_loss = self.distance(quantized,inputs.detach())
            #print(torch.mean(e_latent_loss))
            #print(torch.mean(q_latent_loss))
            c_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
            quantized = inputs + (quantized - inputs).detach() # it should be replaced by Mobius addition when you later use 'quantized' for hyperbolic space
        else:
            c_loss = torch.tensor(0.0, requires_grad=True).to(device)
            quantized = inputs


        quantized_all = torch.zeros_like(inputs_all)
        quantized_all[mask] = quantized
        quantized_all[~mask] = inputs_invalid

        print(torch.mean(c_loss))
        
        return torch.mean(c_loss), quantized_all
