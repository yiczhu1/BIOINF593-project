#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# -----------------------------
# 0. Environment (optional)
# -----------------------------

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# If you want to restrict to certain GPU: os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1. DRASI model definition
# -----------------------------

class DRASIEncoder(nn.Module):
    def __init__(self, in_dim, hidden_mlp=128, hidden_gnn=128, latent_dim=32):
        super().__init__()
        # MLP on PCA features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, hidden_gnn),
            nn.ReLU(),
        )
        # GNN layers over spatial graph
        self.gnn1 = GraphConv(hidden_gnn, hidden_gnn)
        self.gnn2 = GraphConv(hidden_gnn, hidden_gnn)

        # Heads for VAE parameters
        self.mu_head = nn.Linear(hidden_gnn, latent_dim)
        self.logvar_head = nn.Linear(hidden_gnn, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.mlp(x)
        h = F.relu(self.gnn1(h, edge_index, edge_attr))
        h = F.relu(self.gnn2(h, edge_index, edge_attr))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class DRASIDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dec=128, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dec),
            nn.ReLU(),
            nn.Linear(hidden_dec, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class DRASIModel(nn.Module):
    def __init__(self, in_dim, n_domains, latent_dim=32):
        super().__init__()
        self.encoder = DRASIEncoder(in_dim, latent_dim=latent_dim)
        self.decoder = DRASIDecoder(latent_dim, out_dim=in_dim)
        self.n_domains = n_domains

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


# -----------------------------
# 2. Losses
# -----------------------------

def loss_recon(x, x_recon):
    return F.mse_loss(x_recon, x)


def loss_kl(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def compute_domain_means(z, domain_idx, n_domains):
    """
    z: (N, latent_dim)
    domain_idx: (N,) int
    returns (n_domains, latent_dim)
    """
    latent_dim = z.size(1)
    device = z.device

    sums = torch.zeros(n_domains, latent_dim, device=device)
    counts = torch.zeros(n_domains, device=device)

    sums.index_add_(0, domain_idx, z)
    ones = torch.ones_like(domain_idx, dtype=torch.float, device=device)
    counts.index_add_(0, domain_idx, ones)

    counts = counts.clamp(min=1.0)
    means = sums / counts.unsqueeze(1)
    return means


def loss_ddg(z, domain_idx, DD_edge_index, DD_edge_attr, n_domains, lambda_ddg=1.0):
    """
    z: (N, latent_dim)
    domain_idx: (N,) int
    DD_edge_index: (2, E_D)
    DD_edge_attr: (E_D,)
    """
    device = z.device
    domain_idx = domain_idx.to(device)
    DD_edge_index = DD_edge_index.to(device)
    DD_edge_attr = DD_edge_attr.to(device)

    domain_means = compute_domain_means(z, domain_idx, n_domains)  # (D, latent_dim)

    d_i = DD_edge_index[0]
    d_j = DD_edge_index[1]

    mu_i = domain_means[d_i]
    mu_j = domain_means[d_j]

    diff = mu_i - mu_j
    dist_sq = (diff * diff).sum(dim=1)

    weighted = DD_edge_attr * dist_sq
    return lambda_ddg * weighted.mean()


def total_loss(data, model, lambda_kl=1e-3, lambda_ddg=0.1):
    """
    data: PyG Data
    model: DRASIModel
    """
    x = data.x
    y_domain = data.y_domain
    DD_edge_index = data.DD_edge_index
    DD_edge_attr = data.DD_edge_attr
    n_domains = int(data.n_domains)

    x_recon, mu, logvar, z = model(data)

    L_rec = loss_recon(x, x_recon)
    L_kl = loss_kl(mu, logvar)
    L_DDG = loss_ddg(z, y_domain, DD_edge_index, DD_edge_attr, n_domains)

    L = L_rec + lambda_kl * L_kl + L_DDG
    logs = {
        "loss": float(L.item()),
        "rec": float(L_rec.item()),
        "kl": float(L_kl.item()),
        "ddg": float(L_DDG.item()),
    }
    return L, logs, z


# -----------------------------
# 3. Preprocessing & graphs
# -----------------------------

def preprocess_adata(adata, run_normalization=True, hvgs=2000, n_pcs=35):
    """
    DRASI pre-processing:
    - highly variable genes
    - normalize_total
    - log1p
    - scale
    - PCA → adata.obsm['X_pca']
    """
    if run_normalization:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvgs)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)
    return adata


def build_spatial_graph(coords, k=8):
    """
    coords: (N, 2) numpy array
    returns: edge_index (2, E) tensor, edge_attr (E,) tensor
    """
    n_cells = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    rows, cols, weights = [], [], []
    for i in range(n_cells):
        for j, d in zip(indices[i, 1:], distances[i, 1:]):
            rows.append(i)
            cols.append(j)
            weights.append(np.exp(-d))

    A_spatial = sp.coo_matrix(
        (weights, (rows, cols)), shape=(n_cells, n_cells)
    )
    A_spatial = A_spatial.maximum(A_spatial.T)

    if not isinstance(A_spatial, sp.coo_matrix):
        A_spatial = A_spatial.tocoo()

    edge_index = torch.tensor(
        np.vstack([A_spatial.row, A_spatial.col]), dtype=torch.long
    )
    edge_attr = torch.tensor(A_spatial.data, dtype=torch.float)
    return edge_index, edge_attr


def build_ddg(X_pca, domain_idx, edge_index, n_domains, alpha=0.5, thr=0.1):
    """
    X_pca: (N, n_pcs)
    domain_idx: (N,) int
    edge_index: (2, E)
    returns: DD_edge_index, DD_edge_attr
    """
    eps = 1e-8
    X_pca = np.asarray(X_pca, dtype=float)
    domain_idx = np.asarray(domain_idx, dtype=int)

    DD_counts = np.zeros((n_domains, n_domains), dtype=float)

    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()

    for i_cell, j_cell in zip(rows, cols):
        d_i = domain_idx[i_cell]
        d_j = domain_idx[j_cell]
        DD_counts[d_i, d_j] += 1.0
        DD_counts[d_j, d_i] += 1.0

    DD_phys = DD_counts / (DD_counts.sum(axis=1, keepdims=True) + eps)

    # domain means
    domain_mean = np.zeros((n_domains, X_pca.shape[1]), dtype=float)
    domain_counts = np.zeros(n_domains, dtype=float)
    for cell, d in enumerate(domain_idx):
        domain_mean[d] += X_pca[cell]
        domain_counts[d] += 1.0
    domain_counts[domain_counts == 0] = 1.0
    domain_mean = domain_mean / domain_counts[:, None]

    from numpy.linalg import norm

    DD_sem = np.zeros((n_domains, n_domains), dtype=float)
    for i in range(n_domains):
        for j in range(n_domains):
            vi, vj = domain_mean[i], domain_mean[j]
            DD_sem[i, j] = np.dot(vi, vj) / (norm(vi) * norm(vj) + eps)

    DD_combined = alpha * DD_phys + (1.0 - alpha) * DD_sem

    DD_mask = (DD_combined > thr).astype(float)
    DD_final = DD_combined * DD_mask

    DD_rows, DD_cols = np.where(DD_final > 0)
    DD_weights = DD_final[DD_rows, DD_cols]

    DD_edge_index = torch.tensor(
        np.vstack([DD_rows, DD_cols]), dtype=torch.long
    )
    DD_edge_attr = torch.tensor(DD_weights, dtype=torch.float)

    return DD_edge_index, DD_edge_attr


# -----------------------------
# 4. Main runner (DRASI)
# -----------------------------

def run_drasi(
    input_file,
    output_dir,
    sample,
    nclust,
    domain_col="ground_truth",  # true labels in iSTBench DLPFC
    run_normalization=True,
    hvgs=2000,
    n_pcs=35,
    k_spatial=8,
    latent_dim=32,
    n_epochs=100,
    lambda_kl=1e-3,
    lambda_ddg=0.1,
    seed=0,
):
    """
    Train DRASI on multi-slice data and save result compatible with eval script.

    - input_file: combined multi-slice .h5ad (e.g. Slices_combind_data.h5ad)
    - output_dir: where to save {sample}.h5ad
    - sample: model name (e.g. 'DRASI')
    - nclust: number of domains for KMeans on latent embedding
    - domain_col: column with ground-truth labels (copied to 'original_domain')
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()

    # 1. Load data
    adata = sc.read_h5ad(input_file)

    # Make sure slices exist and are categorical for evaluation
    if "slices" not in adata.obs.columns:
        raise ValueError("adata.obs must contain a 'slices' column for batch labels.")
    adata.obs["slices"] = adata.obs["slices"].astype(str).astype("category")

    # Ground-truth domain column -> original_domain for eval script
    if domain_col not in adata.obs.columns:
        raise ValueError(
            f"True label column '{domain_col}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    adata.obs["original_domain"] = adata.obs[domain_col].astype(str).astype("category")

    # 2. Preprocess (HVG + norm + PCA)
    adata = preprocess_adata(
        adata,
        run_normalization=run_normalization,
        hvgs=hvgs,
        n_pcs=n_pcs,
    )
    X_pca = adata.obsm["X_pca"]

    # 3. Spatial coordinates
    if "spatial" in adata.obsm_keys():
        coords = adata.obsm["spatial"]
    elif "X_spatial_coords" in adata.obsm_keys():
        coords = adata.obsm["X_spatial_coords"]
    elif "spatial_coords" in adata.obsm_keys():
        coords = adata.obsm["spatial_coords"]
    else:
        # fallback
        coords = adata.obs[["X", "Y"]].values
    coords = coords.astype(float)

    # 4. Domains for DDG (from original_domain)
    domains = adata.obs["original_domain"].astype("category")
    domain_idx = domains.cat.codes.to_numpy()
    n_domains = len(domains.cat.categories)

    # 5. Build spatial graph
    edge_index, edge_attr = build_spatial_graph(coords, k=k_spatial)

    # 6. Build DDG
    DD_edge_index, DD_edge_attr = build_ddg(
        X_pca, domain_idx, edge_index, n_domains, alpha=0.5, thr=0.1
    )

    # 7. Build PyG Data object
    X_tensor = torch.tensor(X_pca, dtype=torch.float)
    data = Data(
        x=X_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.y_domain = torch.tensor(domain_idx, dtype=torch.long)
    data.n_domains = int(n_domains)
    data.DD_edge_index = DD_edge_index
    data.DD_edge_attr = DD_edge_attr

    data = data.to(device)

    # 8. Train DRASI
    in_dim = X_tensor.size(1)
    model = DRASIModel(in_dim=in_dim, n_domains=n_domains, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Using device: {device}")
    print(f"n_cells={adata.n_obs}, n_genes={adata.n_vars}, n_domains={n_domains}")
    print(f"Start training DRASI for {n_epochs} epochs...")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        L, logs, z = total_loss(
            data,
            model,
            lambda_kl=lambda_kl,
            lambda_ddg=lambda_ddg,
        )
        L.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={logs['loss']:.4f} | "
                f"rec={logs['rec']:.4f} | "
                f"kl={logs['kl']:.4f} | "
                f"ddg={logs['ddg']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        _, _, _, z = model(data)
    z_np = z.cpu().numpy()

    # 9. Store embedding in key expected by eval script
    #   → It looks for obsm["X_embedding"]
    adata.obsm["X_embedding"] = z_np

    # 10. Cluster latent space into nclust (predicted domains)
    km = KMeans(n_clusters=nclust, random_state=seed)
    cluster_labels = km.fit_predict(z_np)
    adata.obs["predicted_domain"] = pd.Categorical(cluster_labels)

    # 11. Save result
    out_h5ad = os.path.join(output_dir, f"{sample}.h5ad")
    adata.write(out_h5ad)

    t1 = time.time()
    runtime = t1 - t0

    # Optionally: stats CSV (your eval script doesn’t require it, but nice to have)
    stats_df = pd.DataFrame(
        {
            "Model": [sample],
            "Execution_Time_s": [runtime],
        }
    )
    stats_csv = os.path.join(output_dir, f"{sample}_stats.csv")
    stats_df.to_csv(stats_csv, index=False)

    print(f"Saved result h5ad to: {out_h5ad}")
    print(f"Saved stats to: {stats_csv}")
    print(f"Total runtime: {runtime:.2f} s")


# -----------------------------
# 5. CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run DRASI (multi-slice) and save result for evaluation")
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="Path to combined multi-slice .h5ad (e.g., Slices_combind_data.h5ad)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save {sample}.h5ad for evaluation")
    parser.add_argument("--sample", "-s", type=str, required=True,
                        help="Sample / model name for output file (e.g. DRASI)")
    parser.add_argument("--nclust", "-c", type=int, required=True,
                        help="Number of clusters (domains) for KMeans on latent space")
    parser.add_argument("--domain_col", "-d", type=str, default="ground_truth",
                        help="Column in obs with true domain labels (copied to original_domain)")
    parser.add_argument("--runNormalization", "-n", type=bool, default=True,
                        help="Whether to run HVG + normalization + scaling (default: True)")
    parser.add_argument("--hvgs", type=int, default=2000,
                        help="Number of highly variable genes (default: 2000)")
    parser.add_argument("--n_pcs", type=int, default=35,
                        help="Number of PCA components (default: 35)")
    parser.add_argument("--k_spatial", type=int, default=8,
                        help="k for spatial kNN graph (default: 8)")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent dimension (default: 32)")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--lambda_kl", type=float, default=1e-3,
                        help="KL weight (default: 1e-3)")
    parser.add_argument("--lambda_ddg", type=float, default=0.1,
                        help="DDG weight (default: 0.1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")

    args = parser.parse_args()

    run_drasi(
        input_file=args.input_file,
        output_dir=args.output_dir,
        sample=args.sample,
        nclust=args.nclust,
        domain_col=args.domain_col,
        run_normalization=args.runNormalization,
        hvgs=args.hvgs,
        n_pcs=args.n_pcs,
        k_spatial=args.k_spatial,
        latent_dim=args.latent_dim,
        n_epochs=args.n_epochs,
        lambda_kl=args.lambda_kl,
        lambda_ddg=args.lambda_ddg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()