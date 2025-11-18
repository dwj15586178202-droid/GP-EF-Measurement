#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ef_quick_train_and_interval.py
--------------------------------
从 PCA+GP 的逐帧潜变量 (latent_smooth, latent_std) 还原视频级 φ_E，
训练 Echo-EF 回归器（Ridge/Elastic 或 MLP，可选蒸馏），并用蒙特卡洛传播 GP 不确定性得到 EF 区间。

快速要点：
- --precache_phi 一次性缓存 φ_E 与 EF，之后训练基本秒级启动；
- --fast_baseline ridge 几秒拿到强线性基线；可 --model ridge 直接用它；
- --model mlp + --distill_from ridge：先蒸馏再微调，训练更快更稳；
- 区间用 MC（默认 64；最后出报告时提到 400~1000）。

依赖：numpy pandas scikit-learn tensorflow>=2.9（仅 MLP 时用；如果只用 ridge/elastic，可不装 TF）
"""

import os, json, glob, argparse, logging, math, pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 尝试导入 TensorFlow（仅在 --model mlp 时才真的用到）
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
except Exception:
    TF_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# -------------------- 基础工具 --------------------
def set_seed(seed: int):
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)

def load_id_list(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids

def load_id_to_ef(meta_csv: str) -> Dict[str, float]:
    df = pd.read_csv(meta_csv)
    cols = {c.lower(): c for c in df.columns}
    if "id" not in cols or "ef" not in cols:
        raise ValueError(f"{meta_csv} 必须包含列 ID 和 EF，实际列：{list(df.columns)}")
    id_col = cols["id"]; ef_col = cols["ef"]
    ef = df[ef_col].astype(float).values
    if np.nanmax(ef) <= 1.0 + 1e-8:
        ef = ef * 100.0
    return dict(zip(df[id_col].astype(str).values, ef))

def try_load_s_star(s_star_json: Optional[str]) -> float:
    if not s_star_json:
        return 1.0
    try:
        data = json.load(open(s_star_json, "r", encoding="utf-8"))
        if isinstance(data, dict) and "s_star" in data:
            return float(data["s_star"])
        if isinstance(data, (int, float)):
            return float(data)
    except Exception as e:
        logging.warning(f"读取 s_star 失败（使用1.0）：{e}")
    return 1.0

def load_gp_npz(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    d = np.load(path, allow_pickle=True)
    mu = d["latent_smooth"] if "latent_smooth" in d else d["mu"]
    std = d["latent_std"] if "latent_std" in d else d["std"]
    meta = {}
    if "meta" in d:
        try:
            meta_bytes = d["meta"].tobytes() if hasattr(d["meta"], "tobytes") else d["meta"]
            meta = json.loads(meta_bytes.decode("utf-8")) if isinstance(meta_bytes, (bytes, bytearray)) else dict(d["meta"].item())
        except Exception:
            try:
                meta = dict(d["meta"].item())
            except Exception:
                meta = {}
    return mu, std, meta

def resolve_times(T: int, meta: dict) -> np.ndarray:
    if isinstance(meta, dict):
        if "times" in meta:
            t = np.asarray(meta["times"], dtype=np.float64).reshape(-1)
            if t.shape[0] == T: return t
        if "fps" in meta:
            fps = float(meta["fps"]) if meta["fps"] is not None else 0.0
            if fps > 0: return np.arange(T, dtype=np.float64) / fps
    return np.linspace(0.0, 1.0, T, dtype=np.float64)

def fit_freq_tau(L1: np.ndarray, L2: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    theta = np.unwrap(np.arctan2(L1, L2))
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, theta, rcond=None)[0]
    f = a / (2.0*np.pi)
    tau = 0.0 if abs(f) < 1e-8 else - b / (2.0*np.pi*f)
    return float(f), float(tau)

def aggregate_phi(mu: np.ndarray, std: np.ndarray, t: np.ndarray, eps: float=1e-12) -> np.ndarray:
    """
    mu/std: [T,D]；D>=2
    φ_E = [f, tau, 形状参数(加权均值, D-2维)]
    """
    T, D = mu.shape
    if D < 2: raise ValueError("潜变量维度 D<2，无法拟合 f,tau")
    f, tau = fit_freq_tau(mu[:,0], mu[:,1], t)
    shape = np.zeros(D-2, dtype=np.float64)
    for k in range(2, D):
        w = 1.0 / (np.maximum(std[:,k], 0.0)**2 + eps)
        shape[k-2] = np.sum(w*mu[:,k]) / (np.sum(w) + eps)
    phi = np.concatenate([[f, tau], shape.astype(np.float64)], axis=0)  # [D]
    return phi

def collect_phi_for_ids(ids: List[str], gp_dir: str, id2ef: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, Y, keep = [], [], []
    for vid in ids:
        p = Path(gp_dir) / f"{vid}.npz"
        if not p.exists(): continue
        mu, std, meta = load_gp_npz(str(p))
        T, D = mu.shape
        t = resolve_times(T, meta)
        phi = aggregate_phi(mu, std, t)
        if str(vid) in id2ef:
            X.append(phi.astype(np.float32)); Y.append(float(id2ef[str(vid)])); keep.append(vid)
    if len(X)==0:
        return np.zeros((0,2),dtype=np.float32), np.zeros((0,),dtype=np.float32), []
    return np.stack(X,0), np.array(Y,dtype=np.float32), keep

def save_scaler(path: str, scaler: StandardScaler):
    np.savez_compressed(path, mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64))
def load_scaler(path: str) -> StandardScaler:
    d = np.load(path)
    sc = StandardScaler()
    sc.mean_ = d["mean"].astype(np.float64)
    sc.scale_ = d["scale"].astype(np.float64)
    sc.var_ = sc.scale_**2
    return sc

def pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    if a.size<2: return float("nan")
    return float(np.corrcoef(a, b)[0,1])

def evaluate(y_true, y_pred_det, y_pred_mc, lo, hi, alpha):
    mae_det = float(mean_absolute_error(y_true, y_pred_det))
    rmse_det = float(np.sqrt(np.mean((y_true - y_pred_det)**2)))
    r2_det   = float(r2_score(y_true, y_pred_det))
    r_det    = pearsonr_np(y_true, y_pred_det)

    mae_mc = float(mean_absolute_error(y_true, y_pred_mc))
    rmse_mc = float(np.sqrt(np.mean((y_true - y_pred_mc)**2)))
    r2_mc   = float(r2_score(y_true, y_pred_mc))
    r_mc    = pearsonr_np(y_true, y_pred_mc)

    coverage = float(np.mean((y_true>=lo)&(y_true<=hi)))
    width    = float(np.mean(hi-lo))
    return dict(N=int(y_true.size), Alpha=float(alpha),
                MAE_det=mae_det, RMSE_det=rmse_det, R2_det=r2_det, r_det=r_det,
                MAE_mc=mae_mc, RMSE_mc=rmse_mc, R2_mc=r2_mc, r_mc=r_mc,
                Coverage=coverage, Width=width)

# -------------------- 训练器：Ridge / Elastic / MLP --------------------
def train_ridge(Xtr, Ytr, alpha=1.0):
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(Xtr, Ytr)
    return model

def train_elastic(Xtr, Ytr, alpha=0.01, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0, max_iter=5000)
    model.fit(Xtr, Ytr)
    return model

def build_mlp(input_dim: int, hidden: List[int], dropout: float, l2w: float):
    if not TF_AVAILABLE:
        raise RuntimeError("未安装 TensorFlow；若只用 ridge/elastic，请设置 --model ridge/elastic")
    x = layers.Input(shape=(input_dim,), dtype=tf.float32)
    h = x
    for units in hidden:
        h = layers.Dense(units, activation="relu",
                         kernel_regularizer=regularizers.l2(l2w) if l2w>0 else None)(h)
        if dropout>0: h = layers.Dropout(dropout)(h)
    y = layers.Dense(1, activation="linear")(h)  # 输出 EF 百分比
    m = models.Model(x, y)
    return m

# -------------------- MC 区间（对任意模型通用） --------------------
def predict_det(model, Xn):
    # 兼容 sklearn 或 keras
    if TF_AVAILABLE and hasattr(model, "predict") and isinstance(model, tf.keras.Model):
        return model.predict(Xn, verbose=0).ravel()
    else:
        return model.predict(Xn).ravel()

def predict_mc_for_ids(ids: List[str], gp_dir: str, scaler: StandardScaler,
                       model, s_star: float, mc_samples: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    y_mc_mean, lo, hi = [], [], []
    for vid in ids:
        p = Path(gp_dir)/f"{vid}.npz"
        mu, std, meta = load_gp_npz(str(p))
        T, D = mu.shape
        t = resolve_times(T, meta)

        # MC
        samples = np.empty((mc_samples,), dtype=np.float64)
        for i in range(mc_samples):
            eps = rng.standard_normal(size=mu.shape)
            Ls = mu + (s_star * std) * eps
            phi = aggregate_phi(Ls, std, t)
            x = scaler.transform(phi[None,:]).astype(np.float32)
            samples[i] = float(predict_det(model, x)[0])
        y_mc_mean.append(float(np.mean(samples)))
        ql = float(np.quantile(samples, (1-alpha)/2))
        qh = float(np.quantile(samples, 1-(1-alpha)/2))
        lo.append(ql); hi.append(qh)
    return np.array(y_mc_mean), np.array(lo), np.array(hi)

# -------------------- 主程序 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp_dir", required=True, type=str)
    ap.add_argument("--meta_csv", required=True, type=str)
    ap.add_argument("--train_ids", required=True, type=str)
    ap.add_argument("--val_ids", required=True, type=str)
    ap.add_argument("--test_ids", required=True, type=str)
    ap.add_argument("--work_dir", required=True, type=str)

    # 训练/模型
    ap.add_argument("--model", choices=["ridge","elastic","mlp"], default="ridge",
                    help="选择快速基线或 MLP")
    ap.add_argument("--fast_baseline", choices=["none","ridge","elastic"], default="none",
                    help="（可选）用线性基线先拿到好解，用于蒸馏")
    ap.add_argument("--distill_from", choices=["none","ridge","elastic"], default="none",
                    help="MLP 先拟合基线的输出（蒸馏），再用真值微调")
    ap.add_argument("--hidden", type=str, default="64,32", help="MLP 隐藏层，如 '64,32'")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=6)

    # 不确定性与评估
    ap.add_argument("--s_star_json", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--mc_samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_pred_csv", action="store_true")

    # 预缓存
    ap.add_argument("--precache_phi", action="store_true", help="缓存 φ_E 与标签，加速后续训练")

    args = ap.parse_args()
    set_seed(args.seed)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    # 1) 读取标签与 split
    id2ef = load_id_to_ef(args.meta_csv)
    train_ids = load_id_list(args.train_ids)
    val_ids   = load_id_list(args.val_ids)
    test_ids  = load_id_list(args.test_ids)

    # 2) 构造 φ_E（可缓存）
    cache_dir = Path(args.work_dir)/"phi_cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    def build_or_load(split_name, ids):
        cache_path = cache_dir/f"{split_name}_phi.npz"
        if cache_path.exists():
            d = np.load(cache_path, allow_pickle=True)
            return d["X"], d["Y"], list(d["ids"])
        X, Y, kept = collect_phi_for_ids(ids, args.gp_dir, id2ef)
        if args.precache_phi:
            np.savez_compressed(cache_path, X=X, Y=Y, ids=np.array(kept))
            logging.info(f"缓存 {split_name} φ_E → {cache_path}")
        return X, Y, kept

    Xtr, Ytr, kept_tr = build_or_load("train", train_ids)
    Xva, Yva, kept_va = build_or_load("val",   val_ids)
    Xte, Yte, kept_te = build_or_load("test",  test_ids)

    logging.info(f"特征维度={Xtr.shape[1] if Xtr.size else 'NA'} | train/val/test={len(kept_tr)}/{len(kept_va)}/{len(kept_te)}")
    if len(kept_tr)==0:
        raise RuntimeError("训练集为空；检查 gp_dir / train_ids / meta_csv 的 ID 是否一致")

    # 3) 标准化（fit on train）
    scaler_path = str(Path(args.work_dir)/"scaler_latent.npz")
    scaler = StandardScaler().fit(Xtr)
    save_scaler(scaler_path, scaler)
    Xtr_n = scaler.transform(Xtr).astype(np.float32)
    Xva_n = scaler.transform(Xva).astype(np.float32)
    Xte_n = scaler.transform(Xte).astype(np.float32)

    # 4) （可选）先训练一个快速基线（几秒搞定）
    teacher = None
    if args.fast_baseline != "none":
        kind = args.fast_baseline
        logging.info(f"训练快速基线（{kind}）...")
        if kind=="ridge":
            teacher = train_ridge(Xtr_n, Ytr, alpha=1.0)
        else:
            teacher = train_elastic(Xtr_n, Ytr, alpha=0.01, l1_ratio=0.5)
        val_pred = teacher.predict(Xva_n).ravel()
        logging.info(f"[FAST-{kind}] val MAE={mean_absolute_error(Yva, val_pred):.3f}")

    # 5) 训练目标模型（ridge/elastic 或 mlp）
    model_path = ""
    if args.model in ["ridge","elastic"]:
        if args.model=="ridge":
            model = train_ridge(Xtr_n, Ytr, alpha=1.0)
            model_path = str(Path(args.work_dir)/"ridge_model.joblib")
        else:
            model = train_elastic(Xtr_n, Ytr, alpha=0.01, l1_ratio=0.5)
            model_path = str(Path(args.work_dir)/"elastic_model.joblib")
        joblib.dump(model, model_path)
        logging.info(f"线性模型已保存 → {model_path}")

    else:
        if not TF_AVAILABLE:
            raise RuntimeError("选择了 --model mlp，但未安装 TensorFlow")
        hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
        model = build_mlp(Xtr_n.shape[1], hidden, args.dropout, args.l2)
        model.compile(optimizer=optimizers.Adam(args.lr), loss="mae", metrics=["mse"])
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, args.patience//2), min_lr=1e-6),
            callbacks.ModelCheckpoint(str(Path(args.work_dir)/"ef_mlp.h5"),
                                      monitor="val_loss", save_best_only=True, save_weights_only=True),
        ]
        # 蒸馏（可选）：先拟合 teacher 的输出
        if args.distill_from!="none" and teacher is not None:
            logging.info(f"先做蒸馏（拟合 {args.distill_from} 的软标签）...")
            Ytr_soft = teacher.predict(Xtr_n).ravel()
            Yva_soft = teacher.predict(Xva_n).ravel()
            model.fit(Xtr_n, Ytr_soft, validation_data=(Xva_n, Yva_soft),
                      epochs=20, batch_size=args.batch_size, verbose=2, callbacks=cbs)
        logging.info("用真值 EF 微调 ...")
        model.fit(Xtr_n, Ytr, validation_data=(Xva_n, Yva),
                  epochs=args.epochs, batch_size=args.batch_size, verbose=2, callbacks=cbs)
        model.load_weights(str(Path(args.work_dir)/"ef_mlp.h5"))
        model_path = str(Path(args.work_dir)/"ef_mlp.h5")
        logging.info(f"MLP 最佳权重已保存 → {model_path}")

    # 6) 评估（Det + MC 区间）
    s_star = try_load_s_star(args.s_star_json)
    logging.info(f"使用 s*={s_star:.4f}, MC={args.mc_samples}, alpha={args.alpha}")

    # Det
    y_det_val = predict_det(model, Xva_n)
    y_det_te  = predict_det(model, Xte_n)

    # MC
    y_mc_mean_val, lo_val, hi_val = predict_mc_for_ids(kept_va, args.gp_dir, scaler, model, s_star, args.mc_samples, args.alpha, args.seed)
    y_mc_mean_te,  lo_te,  hi_te  = predict_mc_for_ids(kept_te, args.gp_dir, scaler, model, s_star, args.mc_samples, args.alpha, args.seed)

    # 指标
    m_val = evaluate(Yva, y_det_val, y_mc_mean_val, lo_val, hi_val, args.alpha)
    m_te  = evaluate(Yte, y_det_te,  y_mc_mean_te,  lo_te,  hi_te,  args.alpha)
    logging.info("[VAL] " + ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k,v in m_val.items()]))
    logging.info("[TEST] " + ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k,v in m_te.items()]))

    # 保存结果
    out = dict(val=m_val, test=m_te,
               settings=vars(args),
               model_path=model_path, scaler_path=scaler_path)
    with open(Path(args.work_dir)/"eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if args.save_pred_csv:
        pd.DataFrame(dict(ID=kept_va, EF_true=Yva, EF_det=y_det_val, EF_mc=y_mc_mean_val,
                          CI_low=lo_val, CI_high=hi_val)).to_csv(Path(args.work_dir)/"pred_val.csv", index=False)
        pd.DataFrame(dict(ID=kept_te, EF_true=Yte, EF_det=y_det_te, EF_mc=y_mc_mean_te,
                          CI_low=lo_te, CI_high=hi_te)).to_csv(Path(args.work_dir)/"pred_test.csv", index=False)
        logging.info("已保存 pred_val.csv / pred_test.csv")

if __name__ == "__main__":
    main()
