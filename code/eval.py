# evaluation
import torch
from model import embedding_distance, lineage_distance
from model import flat_clustering, get_leaves_dict, get_leaves_lca

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix as MCM
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import random

import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb
from adjustText import adjust_text
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from COSNE.htsne_impl import TSNE as hTSNE
import COSNE.hyptorch.pmath as pmath
from scipy.sparse.csgraph import connected_components
from umap import UMAP


def tSNE_fit(X, df):
    data = TSNE(n_components=2, random_state=42).fit_transform(X)
    df2 = pd.DataFrame(data, columns=['tSNE 1', 'tSNE 2'])
    df = pd.concat([df, df2], axis=1)
    return df

def htSNE_fit(X, df):
    if X.shape[1] >= 100:
        embeddings = PCA(n_components=20, random_state=42).fit_transform(X)
    else:
        embeddings = X
    embeddings = 0.95 * embeddings / np.max(np.sqrt(np.sum(embeddings ** 2, axis=1))) # Euclidian to Pincare disk

    learning_rate = 5.0
    learning_rate_for_h_loss = 0.1
    perplexity = 20
    early_exaggeration = 1.0
    student_t_gamma = 0.1
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True,
                   metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, 
                   student_t_gamma=student_t_gamma, learning_rate=learning_rate, 
                   n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()
    CO_SNE_embeddings = co_sne.fit_transform(dists, embeddings.numpy())
    df2 = pd.DataFrame(CO_SNE_embeddings, columns=['CO-SNE 1', 'CO-SNE 2'])
    df = pd.concat([df, df2], axis=1)
    return df

def UMAP_fit(X, df):
    data = UMAP(n_components=2, random_state=42).fit_transform(X)
    df2 = pd.DataFrame(data, columns=['UMAP 1', 'UMAP 2'])
    df = pd.concat([df, df2], axis=1)
    return df

def UMAP_fit_hyp(X_dist_trn_trn, X_dist_tst_trn, df):
    mapper = UMAP(output_metric='hyperboloid', random_state=42, metric='precomputed').fit(X_dist_trn_trn)
    if X_dist_tst_trn is None:
        z = np.sqrt(1 + np.sum(mapper.embedding_**2, axis=1, keepdims=True))
        data = mapper.embedding_ / (z+1)
    else:
        data_ = mapper.transform(X_dist_tst_trn)
        z = np.sqrt(1 + np.sum(data_**2, axis=1, keepdims=True))
        data = data_ / (z+1)
    df2 = pd.DataFrame(data, columns=['UMAP Poincare 1', 'UMAP Poincare 2'])
    df = pd.concat([df, df2], axis=1)
    return df

def supervised_UMAP_fit(X, df, dist_X, Y):
    umap = UMAP(n_components=2, random_state=42, target_metric='precomputed').fit(X, dist_X)
    data = umap.transform(Y)
    df2 = pd.DataFrame(data, columns=['Supervised UMAP 1', 'Supervised UMAP 2'])
    #data = umap.transform(X) # for later calculation of metrics
    #df3 = pd.DataFrame(data, columns=['Supervised UMAP 1 trn', 'Supervised UMAP 2 trn'])
    df = pd.concat([df, df2], axis=1)
    return df

#def poincare_disk_plot(df, plot_stat, nrows, ncols, index):
#    ax = plt.subplot(nrows, ncols, index)
#    circle1 = plt.Circle(xy=(0,0), radius=1, color='black', fill=False)
#    ax = sns.scatterplot(df, x='Poincare 1', y='Poincare 2', ax=ax, **plot_stat)
#    ax.add_patch(circle1)
#    ax.set(xlim=(-1.5,1.5))
#    ax.set(ylim=(-1.5,1.5))

def get_df(X, taxids, latent_space, wo_enc=False, X_trn=None, X_trn_dist=None, X_dist_trn_trn=None, X_dist_tst_trn=None, min_ids=None):
    df_dic = {'taxid': taxids}
    #print(taxids)
    if min_ids is not None:
        df_dic.update(min_ids=min_ids)
    #print(len(taxids))
    #print(X.shape)
    #print(taxids)
    df = pd.DataFrame(df_dic)
    x_names, y_names = [], []
    if latent_space == 'Euclid':
        if X.shape[1] > 2:
            df = tSNE_fit(X, df)
            x_names.append('tSNE 1')
            y_names.append('tSNE 2')
        elif X.shape[1] == 2:
            df2 = pd.DataFrame(X, columns=['1', '2'])
            df = pd.concat([df, df2], axis=1)
            x_names.append('1')
            y_names.append('2')
        if wo_enc:
            df = htSNE_fit(X, df)
            x_names.append('CO-SNE 1')
            y_names.append('CO-SNE 2')
            #print(df.head(5))
            df = UMAP_fit(X, df)
            x_names.append('UMAP 1')
            y_names.append('UMAP 2')
            if X_trn is not None and X_trn_dist is not None:
                df = supervised_UMAP_fit(X_trn, df, X_trn_dist, X)
                x_names.append('Supervised UMAP 1')
                y_names.append('Supervised UMAP 2')

    elif latent_space == 'Lorentz':
        if X.shape[1] > 3:
            df = UMAP_fit_hyp(X_dist_trn_trn, X_dist_tst_trn, df)
            x_names.append('UMAP Poincare 1')
            y_names.append('UMAP Poincare 2')
        elif X.shape[1] == 3:
            df2 = pd.DataFrame(X, columns=['0', '1', '2'])
            df = pd.concat([df, df2], axis=1)
            df['Poincare 1'] = df['1']/(df['0'] + 1) # Lorentz to Poincare disk
            df['Poincare 2'] = df['2']/(df['0'] + 1)
            x_names.append('Poincare 1')
            y_names.append('Poincare 2')
    elif latent_space in ['GaussianManifold', 'GaussianManifoldL']:
        #assert len(X.shape) == 3 and X.shape[2] == 2
        #for index in range(X.shape[1]):
        #    x = X[:,index,0]
        #    y = np.exp(X[:,index,1]/2) # logvar to sigma
        #    temp = x ** 2 + y ** 2
        #    denominator = temp + 2 * y + 1
        #    x_ = (temp - 1) / denominator # poincare half to poincare disk using Cayley transform
        #    y_ = (-2 * x) / denominator
        #    df2 = pd.DataFrame(np.stack([x_, y_], 1), columns=['Poincare 1 (Element {})'.format(index+1), 'Poincare 2 (Element {})'.format(index+1)])
        #    df = pd.concat([df, df2], axis=1)
        #    x_names.append('Poincare 1 (Element {})'.format(index+1))
        #    y_names.append('Poincare 2 (Element {})'.format(index+1))
        df = UMAP_fit_hyp(X_dist_trn_trn, X_dist_tst_trn, df)
        x_names.append('UMAP Poincare 1')
        y_names.append('UMAP Poincare 2')
    return df, x_names, y_names

def subfig_plot(df, x_name, y_name, plot_stat, ax, label_name=''):
    if ('Poincare' in x_name and 'Poincare' in y_name) or ('CO-SNE' in x_name and 'CO-SNE' in y_name):
        circle1 = plt.Circle(xy=(0,0), radius=1, color='black', fill=False)
        ax.add_patch(circle1)
        ax.set(xlim=(-1.5,1.5))
        ax.set(ylim=(-1.5,1.5))
        #print(df.head(5))
        #print(plot_stat)
        #print(df[plot_stat['hue']].unique())
    #plot_stat = {}
    ax = sns.scatterplot(df, x=x_name, y=y_name, ax=ax, **plot_stat)
    if label_name in df.columns:
        names_array = df[label_name].to_numpy()
        u, indices = np.unique(names_array, return_index=True)
        texts = [ax.text(df.at[ind, x_name], df.at[ind, y_name], '{}'.format(names_array[ind]), ha='center', va='center', fontsize='x-small') for ind in indices if names_array[ind] != 'no info']
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.3))

    if 'min_ids' in df.columns:
        x1 = df[x_name].to_numpy()
        y1 = df[y_name].to_numpy()
        min_ids = df['min_ids'].to_numpy().astype(int)
        x2 = x1[min_ids]
        y2 = y1[min_ids]
        #mask = (df[plot_stat['style']] == 'more than one').to_numpy()
        #x1 = x1[mask]
        #y1 = y1[mask]
        #x2 = x2[mask]
        #y2 = y2[mask]
        #print(x1)
        #print(y1)
        t = np.linspace(0, 1, 50).reshape((-1,1))
        x_plot = x1 + (x2-x1) * t
        y_plot = y1 + (y2-y1) * t
        ax.plot(x_plot, y_plot, color='k', linewidth=0.2, alpha=0.5)

def create_fig(fig_path, df, x_names, y_names, hue_name, style_name, markers, hue_order=None, label_name='', legend=False, figsize=12):
    x_names = [x_names] if isinstance(x_names, str) else x_names
    y_names = [y_names] if isinstance(y_names, str) else y_names
    nrows = ncols = int(np.ceil(np.sqrt(len(x_names))))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, tight_layout=True, figsize=(figsize,figsize), dpi=300)
    plot_stat = {'hue': hue_name, 'style': style_name, 'markers': markers, 'hue_order': hue_order, 'legend': legend}
    for i, x_name, y_name in zip(range(len(x_names)), x_names, y_names):
        pos_i = int(i/ncols)
        pos_j = i - pos_i * ncols
        subfig_plot(df, x_name, y_name, plot_stat, axes[pos_i,pos_j], label_name=label_name)
    plt.savefig(fig_path)

def create_figs(X, latent_space, fig_path, lineages, names_dic=None, hue_order=None, wo_enc=False, X_trn=None, X_trn_dist=None, X_dist_trn_trn=None, X_dist_tst_trn=None):
    codebook_mask = (lineages[:,0] < 0)
    lineages_nc = lineages[~codebook_mask]
    lineages_c  = lineages[codebook_mask]
    lineages_ncc = torch.tensor(np.concatenate([lineages_nc, lineages_c], 0), dtype=torch.int32)
    lin_dist = lineage_distance(lineages_ncc[:,None], lineages_ncc[None]).fill_diagonal_(100.0)
    min_ids = torch.argmin(lin_dist, dim=1).tolist()
    #print(min_ids)
    #print(lineages_ncc[:,0])

    df, x_name, y_name = get_df(np.concatenate([X[~codebook_mask], X[codebook_mask]], 0), 
                                lineages_nc[:,0].tolist() + lineages_c[:,0].tolist(), latent_space, wo_enc=wo_enc, X_trn=X_trn, X_trn_dist=X_trn_dist,
                                X_dist_trn_trn=X_dist_trn_trn, X_dist_tst_trn=X_dist_tst_trn, min_ids=min_ids)
    pd.set_option('display.max_columns', 60)
    #print(df.head(5))
    #print(df.tail(5))
    rank_names = ['genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
    markers = {'no info': 'X', 'just one': '^', 'more than one': 'o', 'codebook': 'D'}
    images = {}
    for rank_j in range(1,lineages.shape[1]-2):
        rank_name = rank_names[rank_j-1]
        lin_nc = lineages_nc[:,rank_j]
        lin_c  = lineages_c[:,rank_j]
        #print(lin.shape)
        #print(rank_name + '_taxid')
        #print(df.columns)
        df[rank_name+'_taxid'] = np.concatenate([lin_nc, lin_c], 0).astype(str)
        v, inverse, c = np.unique(lin_nc, return_inverse=True, return_counts=True)
        marker_names = np.array(['no info' if v[i] == -1 else ('just one' if c[i] == 1 else 'more than one') for i in range(len(v))])
        marker_names_c = np.array(['codebook' for i in range(len(lin_c))])
        df[rank_name+'_marker'] = np.concatenate([marker_names[inverse], marker_names_c], 0)
        label_name = ''
        if names_dic is not None:
            v, inverse  = np.unique(np.concatenate([lin_nc, lin_c], 0), return_inverse=True)
            group_names = np.array(['no info' if (v[i] == -1 or str(v[i]) not in names_dic) else names_dic[str(v[i])] for i in range(len(v))])
            df[rank_name+'_name'] = group_names[inverse]
            label_name = rank_name+'_name'

        fig_path_rank = os.path.splitext(fig_path)[0] + '_{}.png'.format(rank_name)

        fig_path_rank12 = os.path.splitext(fig_path_rank)[0] + '_s12.png'
        create_fig(fig_path_rank12, df, x_name, y_name, rank_name+'_taxid', rank_name+'_marker', markers, hue_order=hue_order, label_name='', figsize=12)
        images['fig_'+rank_name+'_s12'] = wandb.Image(fig_path_rank12)
        fig_path_rank8 = os.path.splitext(fig_path_rank)[0] + '_s8.png'
        create_fig(fig_path_rank8, df, x_name, y_name, rank_name+'_taxid', rank_name+'_marker', markers, hue_order=hue_order, label_name='', figsize=8)
        images['fig_'+rank_name+'_s8'] = wandb.Image(fig_path_rank8)
        if len(label_name) > 0:
            fig_path_rank_a = os.path.splitext(fig_path)[0] + '_{}_annotated.png'.format(rank_name)
            fig_path_rank_a12 = os.path.splitext(fig_path_rank_a)[0] + '_s12.png'
            create_fig(fig_path_rank_a12, df, x_name, y_name, rank_name+'_taxid', rank_name+'_marker', markers, hue_order=hue_order, label_name=label_name, figsize=12)
            images['fig_'+rank_name+'_annotated_s12'] = wandb.Image(fig_path_rank_a12)
            fig_path_rank_a8 = os.path.splitext(fig_path_rank_a)[0] + '_s8.png'
            create_fig(fig_path_rank_a8, df, x_name, y_name, rank_name+'_taxid', rank_name+'_marker', markers, hue_order=hue_order, label_name=label_name, figsize=8)
            images['fig_'+rank_name+'_annotated_s8'] = wandb.Image(fig_path_rank_a8)

        if isinstance(x_name, list) and isinstance(y_name, list) and len(x_name) > 1 and len(y_name) > 1:
            for i, x_name_i, y_name_i in zip(range(len(x_name)), x_name, y_name):
                fig_path_rank_i = os.path.splitext(fig_path_rank)[0] + '_sub_{}.png'.format(i)
                create_fig(fig_path_rank_i, df, x_name_i, y_name_i, rank_name+'_taxid', rank_name+'_marker', markers, hue_order=hue_order, label_name=label_name)
                images['fig_'+rank_name+f'_sub_{i}'] = wandb.Image(fig_path_rank_i)
    #print(df.head(10))
    #print(df.tail(10))
    return images, df, x_name, y_name

def create_heatmap(dist_X, fig_path, fig_name):
    plt.figure()
    sns.heatmap(dist_X)
    plt.savefig(fig_path)
    return {'fig_heatmap_'+fig_name: wandb.Image(fig_path)}

def get_valid_dist_lin(dist, lineages1, lineages2):
    mask1 = lineages1 >= 0
    mask2 = lineages2 >= 0
    return dist[np.ix_(mask1, mask2)], lineages1[mask1], lineages2[mask2]

def macro_average_precision(dist_trn, dist_tst, lineages_trn, lineages_tst, epoch = 0, wo_enc = False, X_trn = None, X_tst = None):
    assert lineages_trn.shape[1] == 9 # assumption: species, genus, ..., superkingdom, no rank, no rank
    mavep = {}
    rank_names = ['genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
    knn_ks = [1,3,5]
    max_k = max(knn_ks)
    for rank_j in range(1,lineages_trn.shape[1]-2):
        classifiers = {
          'KNN': KNeighborsClassifier(n_neighbors=max_k, metric='precomputed')
        }
        if wo_enc and X_trn is not None and X_tst is not None and lineages_tst is not None:
            n_trn = X_trn.shape[0]
            #classifiers.update({
            #    'RandomForest': RandomForestClassifier(random_state=42),
            #    'GradientBoosting': GradientBoostingClassifier(random_state=42),
            #    'BaggedDecisionTree': BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators = 500, random_state=42),
            #    'SubspaceKNN': BaggingClassifier(KNeighborsClassifier(n_neighbors=2), 
            #                                     n_estimators = 100, max_features=n_trn/(n_trn/2+np.sqrt(n_trn)), random_state=42)
            #})
            X_trn_j = X_trn[lineages_trn[:,rank_j] >= 0]
            X_tst_j = X_tst[lineages_tst[:,rank_j] >= 0]
        dist_trn_trn, lin_trn, _ = get_valid_dist_lin(dist_trn, lineages_trn[:,rank_j], lineages_trn[:,rank_j])
        if isinstance(dist_tst, np.ndarray) and isinstance(lineages_tst, np.ndarray):
            #print('ok1')
            dist_tst_trn, lin_tst, _ = get_valid_dist_lin(dist_tst, lineages_tst[:,rank_j], lineages_trn[:,rank_j])
        else:
            #print('ok2')
            dist_tst_trn = None
            lin_tst = None
        if epoch == 1:
            if isinstance(dist_tst, np.ndarray) and isinstance(lineages_tst, np.ndarray):
                v_trn = np.unique(lin_trn)
                v_tst = np.unique(lin_tst)
                mavep_max = 0
                for taxid in v_tst:
                    mavep_max += (1 if taxid in v_trn else 0)
                mavep_max /= len(v_tst)
                #print(rank_names[rank_j-1], mavep_max, len(v_tst))
                mavep['{}_max'.format(rank_names[rank_j-1])] = mavep_max
            else:
                _, c_trn = np.unique(lin_trn, return_counts=True)
                mavep_max = 0
                for c in c_trn:
                    mavep_max += (1 if c > 1 else 0)
                mavep_max /= len(c_trn)
                #print(rank_names[rank_j-1], mavep_max, len(v_tst))
                mavep['{}_max'.format(rank_names[rank_j-1])] = mavep_max

        y_true = lin_tst if lin_tst is not None else lin_trn
        #print('y_true: ', y_true.shape)
        #if dist_tst_trn is not None:
        #    print(dist_tst_trn.shape)
        taxids, inv, counts = np.unique(lin_trn, return_inverse=True, return_counts=True)
        n_classes_trn = len(taxids)
        total_counts = sum(counts)
        weights = {taxid: count/total_counts for taxid, count in zip(taxids, counts)}
        for name, classifier in classifiers.items():
            if name == 'KNN':
                classifier.fit(coo_matrix(dist_trn_trn), lin_trn)
                #print(type(dist_trn_tst))
                neigh_dist, neigh_ind = classifier.kneighbors(coo_matrix(dist_tst_trn)) if isinstance(dist_tst_trn, np.ndarray) else classifier.kneighbors()
                lin_trn_w = np.array([weights[taxid] for taxid in taxids])[inv] # taxid -> its weight
                lin_trn_p = inv # taxid -> its position in taxids
                pred_scores = np.zeros((y_true.shape[0], n_classes_trn))
                for k in range(1, max_k+1):
                    #print(pred_scores.shape)
                    #print(neigh_ind.shape)
                    pred_scores[np.arange(y_true.shape[0]),lin_trn_p[neigh_ind[:,k-1]]] += lin_trn_w[neigh_ind[:,k-1]]
                    #if epoch == 1:
                    #    print(pred_scores[:10])
                    if k in knn_ks:
                        y_pred_p = np.argmax(pred_scores,axis=1) # majority voting
                        y_pred = taxids[y_pred_p]
                        #p = precision_score(y_true, y_pred, average='macro') # precision_score calculates scores for the mixed class of y_true and y_pred
                        mcms = MCM(y_true, y_pred, labels=np.unique(y_true)) # scores for the class of y_true
                        p = np.mean(mcms[:,1,1]/np.maximum(mcms[:,1,1]+mcms[:,0,1], 1))
                        mavep['{}_{}_{}'.format(rank_names[rank_j-1],name,k)] = p
            else:
                classifier.fit(X_trn_j, lin_trn)
                y_pred = classifier.predict(X_tst_j)
                mavep['{}_{}'.format(rank_names[rank_j-1],name)] = precision_score(y_true, y_pred, average='macro')
                
    
    return {'map_'+k:v for k, v in mavep.items()}

#def get_leaves_dict(z):
#    # dictonary: each node (internal/external) to all the descendant leaves
#    n = len(z)+1
#    d = {k: {k} for k in range(n)}
#    for i in range(n,2*n-1):
#        d[i] = d[z[i-n,0]] | d[z[i-n,1]]
#    return d

#def get_leaves_lca(d, i, j):
#    n = int((len(d)+1)/2)
#    for k in range(n,2*n-1):
#        if {i,j} <= d[k]:
#             return d[k]

#def flat_clustering(lineages):
#    assert lineages.shape[1] == 9 # assumption: species, genus, ..., superkingdom, no rank, no rank
#    fc_result = {}
#    rank_names = ['genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
#    for rank_j in range(1,lineages.shape[1]-2):
#        d = defaultdict(set)
#        d['no_info'] = set()
#        for sample_i in range(lineages.shape[0]):
#            if lineages[sample_i, rank_j] >= 0:
#                d[lineages[sample_i, rank_j]] = d[lineages[sample_i, rank_j]] | {sample_i}
#            else:
#                d['no_info'] = d['no_info'] | {sample_i}
#        fc_result[rank_names[rank_j-1]] = d
#    return fc_result

def dendrogram_purity(condense_dist_mat, lineages):
    Z = linkage(condense_dist_mat, 'ward')
    d = get_leaves_dict(Z[:,0:2].astype(int))
    # flat clustering of the true tree at each rank
    fc_result = flat_clustering(lineages)
    dp = {}
    for rank_name, flat_clust_dict in fc_result.items():
        frac = 0
        n_pairs = 0
        no_infos = flat_clust_dict.pop('no_info')
        for taxid, s_true in flat_clust_dict.items():
            for i in s_true:
                for j in s_true:
                    # note: the result of get_leaves_lca may contain species which have no information in the rank and thus are excluded from s_pred
                    s_pred = get_leaves_lca(d, i, j) - no_infos
                    frac += len(s_true & s_pred)/max(len(s_pred),1)
            n_pairs += len(s_true) ** 2
        dp[rank_name] = frac / n_pairs
    return {'dp_' + k: v for k, v in dp.items()}

def delta_hyperbolicity(dist_X):
    anchor_i = random.choice(list(range(dist_X.shape[0])))
    gromov_product = (dist_X[:,anchor_i:anchor_i+1] + dist_X[anchor_i:anchor_i+1,:] - dist_X) / 2
    # max-min product and search for minimum delta
    delta = - np.inf
    for i in range(dist_X.shape[0]):
        for j in range(i, dist_X.shape[0]):
            max_min = - np.inf
            for k in range(dist_X.shape[0]):
                temp = min(gromov_product[i,k], gromov_product[k,j])
                if temp > max_min:
                    max_min = temp
            temp = max_min - gromov_product[i,j]
            if temp > delta:
                delta = temp
    return {'deltaH':  2 * delta / np.max(dist_X)}

def spearman_correlation(X, Y):# 1D or 2D array_like
    corr, pvalue = spearmanr(X, Y)
    return {'sp_corr': corr, 'sp_pvalue': pvalue}

def get_eval_metrics(X_emd, lineages, latent_space, epoch, c=None, save_fig_path=None, X_trn_emd = None, lineages_trn = None, batch_size=1000, names_dic=None, hue_order=None, wo_enc=False, X_vis=None, lin_vis=None, df_trn=None):
    if epoch != 150:
        save_fig_path=None
    results = {}
    indicies = list(range(X_emd.shape[0]))
    random.shuffle(indicies)
    N = X_emd.shape[0]
    #if N < 100:
    #    print(X_emd[:10])
    images = {}
    for i in range(0, N, batch_size):
        end = min(i+batch_size, N)
        bs = end - i
        X = X_emd[indicies[i:end]]
        lins = Y = lineages[indicies[i:end]]
        dist_X = embedding_distance(X[:,None], X[None], latent_space=latent_space)
        dist_Y = lineage_distance(Y[:,None], Y[None])
        #print(X[:10,:,0])
        #print(X[:10,:,1])
        #print(dist_X[:10,:10])
        #print(torch.any(torch.isnan(dist_X)))
        pdist_indices = []
        for j in range(X.shape[0]):
            for k in range(j+1, X.shape[0]):
                pdist_indices.append((j,k))
        l, r = map(list, zip(*pdist_indices))
        pdist_X = dist_X[l,r].numpy()
        pdist_Y = dist_Y[l,r].numpy()
        results_sc = spearman_correlation(pdist_X, pdist_Y)
        #print('scorr')
        results_dp = dendrogram_purity(pdist_X, lins.numpy())
        #print('dp')
        results_ap = {}
        imgs_trn = imgs_tst = None
        if X_trn_emd is not None and lineages_trn is not None:
            dist_tst_trn = embedding_distance(X[:,None], X_trn_emd[None], latent_space=latent_space)
            dist_tst_trn = dist_tst_trn.numpy()
            dist_trn_trn = embedding_distance(X_trn_emd[:,None], X_trn_emd[None], latent_space=latent_space)
            dist_trn_trn = dist_trn_trn.numpy()
            lin_trn = lineages_trn.numpy()
            lin_tst = lins.numpy()
            if wo_enc:
                imgs_trn = X_trn_emd.reshape((X_trn_emd.shape[0], -1)).numpy()
                imgs_tst = X.reshape((X.shape[0], -1)).numpy()
        else:
            dist_trn_trn = dist_X.numpy()
            dist_tst_trn = None
            lin_trn = lins.numpy()
            lin_tst = None
        results_ap = macro_average_precision(dist_trn_trn, dist_tst_trn, lin_trn, lin_tst, epoch, wo_enc, imgs_trn, imgs_tst)
        #print('map')
        results_dh = {} #delta_hyperbolicity(dist_X)
        #print('dh')
        results_temp = dict(**results_sc, **results_dp, **results_dh, **results_ap)
        results = {k: results.get(k, 0) + bs * results_temp.get(k, 0) for k in set(results) | set(results_temp)}
        #print(N, bs, results)
        if save_fig_path != None and i == 0:
            fig_path_heatmap_base = os.path.splitext(save_fig_path)[0] + '-heatmap'
            if wo_enc:
                images.update(create_heatmap(dist_X, fig_path_heatmap_base + '-input_images.png', 'input_images'))
                images.update(create_heatmap(dist_Y, fig_path_heatmap_base + '-lineages.png', 'lineages'))
            else:
                images.update(create_heatmap(dist_X, fig_path_heatmap_base + '-embeddings.png', 'embeddings'))
    
    df_result = None
    umap_results = {}
    if save_fig_path != None:
        #if X_vis is not None:
            #print(X_vis.shape)
            #print(lin_vis.dtype)
            #print(X_vis[:10])
            #print(X_vis[-10:])
            #print(lin_vis[:10])
            #print(lin_vis[-10:])
        if wo_enc: # it should be modified for large scale
            if X_trn_emd is not None and lineages_trn is not None:
                X_trn_umap = X_trn_emd.reshape((X_trn_emd.shape[0], -1)).numpy()
                X_trn_dist_umap = lineage_distance(lineages_trn[:,None], lineages_trn[None]).numpy()
            else:
                X_trn_umap = X_emd.reshape((X_emd.shape[0], -1)).numpy()
                X_trn_dist_umap = lineage_distance(lineages[:,None], lineages[None]).numpy()
        else:
            X_trn_umap = X_trn_dist_umap = None

        X_dist_trn_trn = X_dist_tst_trn = None
        if (latent_space == 'Lorentz' and X_emd.shape[1] > 3) or latent_space == 'GaussianManifold':

            if X_trn_emd is not None:
                X_dist_trn_trn = embedding_distance(X_trn_emd[:,None], X_trn_emd[None], latent_space=latent_space).numpy()
                X_dist_tst_trn = embedding_distance(X_emd[:,None], X_trn_emd[None], latent_space=latent_space).numpy()
            else:
                X_dist_trn_trn = embedding_distance(X_emd[:,None], X_emd[None], latent_space=latent_space).numpy()


        figs, df, x_name, y_name = create_figs(X_emd if X_vis is None else X_vis, latent_space, save_fig_path,
                                               lineages.numpy() if lin_vis is None else lin_vis.numpy(),
                                               names_dic=names_dic, hue_order=hue_order, wo_enc=wo_enc, X_trn=X_trn_umap, X_trn_dist=X_trn_dist_umap,
                                               X_dist_trn_trn=X_dist_trn_trn, X_dist_tst_trn=X_dist_tst_trn)
        images.update(figs)
        if wo_enc:
            df_result = df if df_trn is None else None
            embeddings = df[['Supervised UMAP 1', 'Supervised UMAP 2']].to_numpy()
            dist_embeddings = pairwise_distances(embeddings)
            dist_lineages = lineage_distance(lineages[:,None], lineages[None]).numpy()
            r,c = np.triu_indices(dist_embeddings.shape[0],1)
            pdist_embeddings = dist_embeddings[r,c]
            pdist_lineages = dist_lineages[r,c]
            umap_results_sc = spearman_correlation(pdist_embeddings, pdist_lineages)
            #print('scorr')
            umap_results_dp = dendrogram_purity(pdist_embeddings, lineages.numpy())
            embeddings_trn = df_trn[['Supervised UMAP 1', 'Supervised UMAP 2']].to_numpy() if df_trn is not None else embeddings
            is_test = X_trn_emd is not None and lineages_trn is not None
            dist_embeddings_trn = pairwise_distances(embeddings_trn) if is_test else pairwise_distances(embeddings)
            dist_embeddings_tst = pairwise_distances(embeddings, embeddings_trn) if is_test else None
            umap_lin_trn = lineages_trn.numpy() if is_test else lineages.numpy()
            umap_lin_tst = lineages.numpy()     if is_test else None 
            #if dist_embeddings_trn is not None:
            #    print(dist_embeddings_trn.shape)
            #    print(df.head(5))
            #    print(df.shape[0])
            #    if df_trn is not None:
            #        print(df_trn.head(5))
            #        print(df_trn.shape[0])
            #    if dist_embeddings_tst is not None:
            #        print(dist_embeddings.shape)
            #    if umap_lin_trn is not None:
            #        print(umap_lin_trn.shape[0])
            #    if umap_lin_tst is not None:
            #        print(umap_lin_tst.shape[0])
            umap_results_ap = macro_average_precision(dist_embeddings_trn, dist_embeddings_tst, umap_lin_trn, umap_lin_tst)
            umap_results = dict(**umap_results_sc, **umap_results_dp, **umap_results_ap)
            umap_results = {k + '-supervisedUMAP': v for k,v in umap_results.items()}

    return {**{k: v/N for k,v in results.items()}, **images, **umap_results}, df_result

if __name__ == '__main__':
    # for script execution
    import argparse
    from model import CNN, VAE
    from data_loading import FastaData, my_collate_fn
    from torch.utils.data import DataLoader

    run = wandb.init(
            project='taxonomic-classification-p3',
            name='eval',
        )

    parser = argparse.ArgumentParser(description='<Representation learning for the whole genome taxonomic classification (evaluation)>')

    # model info
    parser.add_argument('--model',  default="Encoder", type=str, required=True, choices=['Encoder', 'VAE', 'VQEncoder', 'VQVAE'], help='encoding model architecture')
    parser.add_argument('--model_backbone',  default="CNN", type=str, required=True, choices=['CNN'], help='encoding backbone')
    parser.add_argument('--model_checkpoint',   default="",  type=str, required=True, help='checkpoint path for the model you want to test')
    parser.add_argument('--kmer',       default=5,       type=int, required=True, help='kmer length')
    parser.add_argument('--latent_space', default='Euclid', type=str, required=False, choices=['GaussianManifold', 'Lorentz', 'Euclid'], help='latent space to embed data points into')
    parser.add_argument('--latent_channels', default=512, type=int, required=False, help='the size of hidden dimension (+1 if hyperbolic)')

    # data related input
    parser.add_argument('--species_fa',   default="",  type=str, required=False, help='Species fasta files')
    parser.add_argument('--species_fcgr_np',   default="",  type=str, required=False, help='Species FCGR images preprocessed and saved either in .npy or .npz format')
    parser.add_argument('--species_fcgr_id',   default="",  type=str, required=False, help='Species FCGR images preprocessed and saved in .josn format')
    parser.add_argument('--acc2id',   default="",  type=str, required=False, help='Accession number to TaxID dictionary')
    parser.add_argument('--id2acc',   default="",  type=str, required=False, help='TaxID to list of accession numbers dictionary')
    parser.add_argument('--scientific_names', default="",  type=str, required=True, help='TaxID to scientific name dictionary')
    parser.add_argument('--lineages', default="",  type=str, required=True, help='TaxID to lineage list dictionary')

    parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
    parser.add_argument('--workers',     default=48,       type=int, required=False, help='number of worker for data loading')
    parser.add_argument('--device', default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')

    args = parser.parse_args()
    
    # model loading
    m_state_dict = torch.load(args.model_checkpoint)
    if "Encoder" in args.model and args.model_backbone == 'CNN':
        model = CNN(args.kmer, latent_channels = args.latent_channels, latent_space = args.latent_space).to(args.device)
    elif 'VAE' in args.model and args.model_backbone == 'CNN':
        model = VAE(args.kmer, latent_channels = args.latent_channels, latent_space = args.latent_space).to(args.device)
    model.load_state_dict(m_state_dict)

    # data loading
    data_set = {
          'species_fa': args.species_fa,
          'species_fcgr_np': args.species_fcgr_np,
          'species_fcgr_id': args.species_fcgr_id,
          'acc2id': args.acc2id,
          'id2acc': args.id2acc,
          'scientific_names': args.scientific_names, 
          'lineages': args.lineages
    }
    data = FastaData(data_set, args.kmer)
    trn_dataset, val_dataset, tst_dataset = data.get_dataset(0), data.get_dataset(1), data.get_dataset(2)
    dataloaders = [DataLoader(trn_dataset, shuffle=True, batch_size=min(len(trn_dataset), args.batch_size), collate_fn=my_collate_fn, num_workers=args.workers),
                   DataLoader(val_dataset, batch_size=min(len(val_dataset), args.batch_size), collate_fn=my_collate_fn, num_workers=args.workers),
                   DataLoader(tst_dataset, batch_size=min(len(tst_dataset), args.batch_size), collate_fn=my_collate_fn, num_workers=args.workers)]
    names_dic = data.get_names_dic()

    for mode in range(3):
        if mode == 0:
            X_trn = None
            lin_trn = None
        X = None
        lin = None
        with torch.no_grad():
            for imgs, lineages in dataloaders[mode]:
                imgs = imgs.to(args.device)
                lineages = lineages.to(args.device)
                outputs = model(imgs)
                c = None
                if 'Encoder' in args.model:
                    embed_mu = outputs
                elif 'VAE' in args.model:
                    rec, embed, embed_mu, embed_logvar = outputs
                X = embed_mu.detach().cpu() if X is None else torch.cat((X,embed_mu.detach().cpu()), 0)
                lin = lineages.detach().cpu() if lin is None else torch.cat((lin,lineages.detach().cpu()), 0)

        save_fig_path = os.path.splitext(args.model_checkpoint)[0] + '-{}-{}.png'.format(['trn', 'val', 'tst'][mode], 'eval')
        results, _ = get_eval_metrics(X, lin, args.latent_space, 150, save_fig_path=save_fig_path, X_trn_emd = X_trn, lineages_trn = lin_trn, names_dic=names_dic)

        wandb.log({['trn_', 'val_', 'tst_'][mode]+key:value for key, value in results.items()})

        if mode == 0:
            X_trn = X
            lin