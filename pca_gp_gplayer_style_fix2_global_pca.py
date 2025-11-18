#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA + GP smoothing for one video, with optional *global PCA* support.
- If --pca_meta is provided, project latents with the saved PCA mean/components
  instead of fitting PCA per-video. This ensures a consistent subspace across videos.
- Otherwise, falls back to per-video PCA (original behavior).

Outputs a single NPZ with:
  latent_smooth (T,D), latent_std (T,D), meta (json string)
Optionally also stores pca_mean/pca_components and Z_std (debug/analysis aid).
"""

import argparse, json, os
import numpy as np
from pathlib import Path
from typing import Tuple

# ---------- utilities ----------
def json_safe(o):
    import numpy as _np
    if isinstance(o, (_np.integer, _np.floating)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return o

# ---------- RBF 核 ----------
def rbf_gram(t: np.ndarray, ell: float) -> np.ndarray:
    t = t.reshape(-1, 1)            # (T,1)
    d2 = (t - t.T) ** 2             # (T,T)
    return np.exp(-0.5 * d2 / (ell ** 2))


def gp_regress_time_series(y: np.ndarray,
                           t: np.ndarray,
                           gamma2: float = 1.0,
                           ell: float = 0.25,
                           sigma2: float = 1e-3,
                           jitter: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    标准 GP 回归（在训练时刻上预测）：
      K_f = γ²·RBF + jitter·I        潜在函数核
      K_y = K_f + σ²·I               观测核（含噪）
      μ    = K_f @ K_y^{-1} y
      Σ_f  = K_f - K_f @ K_y^{-1} @ K_f  （潜在函数方差）
      Σ_obs= Σ_f + σ²·I                 （观测方差）

    返回：(mu, var_f, var_obs)，三者均为 (T,)
    """
    T = len(t)
    R = rbf_gram(t, ell)                    # (T,T)
    K_f = gamma2 * R + jitter * np.eye(T)   # latent kernel
    K_y = K_f + sigma2 * np.eye(T)          # observed kernel

    # Cholesky on K_y
    L = np.linalg.cholesky(K_y)
    # alpha = K_y^{-1} y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = K_f @ alpha                         # (T,)

    # Σ_f 对角：K_f - K_f @ K_y^{-1} @ K_f
    v = np.linalg.solve(L, K_f)              # (T,T)
    var_f = np.clip(np.diag(K_f - v.T @ v), 1e-12, None)  # (T,)
    var_obs = var_f + sigma2                 # 观测方差
    return mu, var_f, var_obs


# ---------- PCA helpers ----------
def fit_pca_per_video(L: np.ndarray,
                      var_thresh: float = 0.98,
                      max_components: int = 12):
    """Per-video PCA (original behavior). Returns Z (T,K), V_K (K,D), mean (D,), ev_ratio_sum, K"""
    T, D = L.shape
    L_mean = L.mean(axis=0, keepdims=True)     # (1,D)
    Lc     = L - L_mean
    U, S, Vt = np.linalg.svd(Lc, full_matrices=False)  # Lc ≈ U S Vt
    expl_var   = (S**2) / max(1, (T - 1))
    expl_ratio = expl_var / max(1e-12, expl_var.sum())
    cum = np.cumsum(expl_ratio)
    K = int(min(max_components, np.searchsorted(cum, var_thresh) + 1))
    K = max(1, K)

    Z = U[:, :K] * S[:K]     # (T,K)
    V_K = Vt[:K, :]          # (K,D)
    evr_sum = float(expl_ratio[:K].sum())
    return Z, V_K, L_mean.squeeze(0), evr_sum, K


def project_with_global_pca(L: np.ndarray, pca_meta_path: str):
    """Project L (T,D) to global PCA space using saved mean/components from json."""
    with open(pca_meta_path, 'r') as f:
        meta = json.load(f)
    mean = np.array(meta["mean"], dtype=np.float32)              # (D,)
    comp = np.array(meta["components"], dtype=np.float32)        # (K,D)
    evr_sum = float(sum(meta.get("explained_variance_ratio", [])))
    K = int(meta.get("n_components", comp.shape[0]))

    Lc = L - mean[None, :]
    Z  = Lc @ comp.T  # (T,K)
    V_K = comp        # (K,D)
    return Z, V_K, mean, evr_sum, K


# ---------- main worker ----------
def run_one_video(latents_dir: str,
                  video_id: str,
                  out_dir: str,
                  var_thresh: float = 0.98,
                  max_components: int = 12,
                  gamma2: float = 1.0,
                  ell: float = 0.25,
                  sigma2: float = 1e-3,
                  jitter: float = 1e-9,
                  pca_meta: str = None,
                  save_pca_fields: bool = True,
                  plot: bool = False):
    lat_path = Path(latents_dir) / f"{video_id}.npy"
    if not lat_path.exists():
        raise FileNotFoundError(f"latent file not found: {lat_path}")
    L = np.load(lat_path)   # (T, D)
    T, D = L.shape

    # ---- PCA：全局 or 单视频 ----
    if pca_meta and os.path.exists(pca_meta):
        Z, V_K, L_mean, evr, K = project_with_global_pca(L, pca_meta)
        pca_mode = "global"
    else:
        Z, V_K, L_mean, evr, K = fit_pca_per_video(L, var_thresh, max_components)
        pca_mode = "per_video"

    # ---- 对每个 PC 做 1D GP（观测方差）----
    t = np.linspace(0., 1., T)
    Z_mu      = np.zeros_like(Z)     # (T,K)
    Z_var_obs = np.zeros_like(Z)     # (T,K)
    for k in range(K):
        mu_k, var_f_k, var_obs_k = gp_regress_time_series(
            y=Z[:, k], t=t, gamma2=gamma2, ell=ell, sigma2=sigma2, jitter=jitter
        )
        Z_mu[:, k]      = mu_k
        Z_var_obs[:, k] = var_obs_k   # 用观测方差以匹配 raw latent 的评估

    # ---- 反投影回 D 维 ----
    # 平滑后的均值 (T,D)
    latent_smooth = (Z_mu @ V_K) + L_mean[None, :]            # (T,D)

    # 线性传播观测方差：Var[L_d] = Σ_k Var[z_k] * W[d,k]^2, W = V_K^T
    W_sq = (V_K.T ** 2)                                      # (D,K)
    latent_var_obs = Z_var_obs @ W_sq.T                      # (T,D)
    latent_var_obs = np.clip(latent_var_obs, 1e-12, None)
    latent_std = np.sqrt(latent_var_obs)                     # (T,D)

    # ---- 保存 ----
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.npz"
    meta = dict(
        n_components=int(K),
        var_thresh=float(var_thresh),
        gamma2=float(gamma2),
        ell=float(ell),
        sigma2=float(sigma2),
        jitter=float(jitter),
        T=int(T), D=int(D),
        std_is_observation_level=True,
        explained_variance=float(evr),
        pca_mode=pca_mode,
        pca_meta=pca_meta if pca_mode == "global" else None,
    )

    save_kwargs = dict(
        latent_smooth=latent_smooth.astype(np.float32),
        latent_std=latent_std.astype(np.float32),
        meta=json.dumps(meta, default=json_safe).encode("utf-8"),
    )
    if save_pca_fields:
        # 便于调试与回溯；如需减小体积可关闭
        save_kwargs.update(
            pca_mean=L_mean.astype(np.float32),
            pca_components=V_K.astype(np.float32),
            Z_std=np.sqrt(np.clip(Z_var_obs, 1e-12, None)).astype(np.float32),
        )

    np.savez_compressed(out_path, **save_kwargs)
    print(f"[OK] {video_id} -> {out_path}  L_smooth:{latent_smooth.shape}  std:{latent_std.shape}  K={K}  PCA={pca_mode}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            i = 0
            Z_std = np.sqrt(np.clip(Z_var_obs, 1e-12, None))
            plt.figure(figsize=(8,3))
            plt.plot(t, Z[:, i], label='raw PC1', alpha=0.5)
            plt.plot(t, Z_mu[:, i], label='GP mean', linewidth=2)
            plt.fill_between(t, Z_mu[:, i]-2*Z_std[:, i], Z_mu[:, i]+2*Z_std[:, i],
                             alpha=0.2, label='±2σ (obs)')
            plt.legend(); plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"[warn] plotting skipped: {e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_dir", required=True, type=str)
    ap.add_argument("--video", required=True, type=str, help="single video id (without .npy)")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--var_thresh", type=float, default=0.98)
    ap.add_argument("--max_components", type=int, default=12)
    ap.add_argument("--gamma2", type=float, default=1.0)
    ap.add_argument("--ell", type=float, default=0.25)
    ap.add_argument("--sigma2", type=float, default=1e-3)
    ap.add_argument("--jitter", type=float, default=1e-9)
    ap.add_argument("--plot", type=int, default=0)
    ap.add_argument("--pca_meta", type=str, default=None,
                    help="Path to global PCA meta json. If provided, skip per-video PCA and use this meta.")
    ap.add_argument("--no_save_pca_fields", action="store_true",
                    help="Don't store pca_mean/components/Z_std in NPZ to save space.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_one_video(
        latents_dir=args.latents_dir,
        video_id=args.video,
        out_dir=args.out_dir,
        var_thresh=args.var_thresh,
        max_components=args.max_components,
        gamma2=args.gamma2,
        ell=args.ell,
        sigma2=args.sigma2,
        jitter=args.jitter,
        pca_meta=args.pca_meta,
        save_pca_fields=(not args.no_save_pca_fields),
        plot=bool(args.plot),
    )
