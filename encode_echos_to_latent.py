import os, sys, glob, yaml
from pathlib import Path
import numpy as np
import tensorflow as tf

# ---------------- GPU 显存自适应（可选但推荐） ----------------
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------------- 基本路径 ----------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "source"))

from source.models.echo_ae import EchoAutoencoderModel

# === 路径配置 ===
EXP_DIR = ROOT / "experiments" / "EchoDynamics" / "20251010-111652"
CKPT_PREFIX = EXP_DIR / "trained_models" / "EAE_best"      # 不带扩展名
CACHE_DIR = Path("/mnt/4DHeartModel/cache1/EchoNet-Dynamic/Videos")
OUT_DIR   = Path("/mnt/4DHeartModel/experiments/EchoNet/latents_raw")
ID_LIST   = Path("/mnt/4DHeartModel/experiments/EchoNet/manifest_fullscan/ids_ok.txt")

# 如需 CAMUS，切换为：
# CACHE_DIR = Path("/mnt/4DHeartModel/cache1/CAMUS/Videos")
# OUT_DIR   = Path("/mnt/4DHeartModel/experiments/CAMUS/latents_raw")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 读取配置并构建模型 ===
CFG = ROOT / "source" / "configs" / "generative_model_no_ef" / "echo.yml"
with CFG.open("r") as f:
    cfg = yaml.safe_load(f)

model = EchoAutoencoderModel(
    model_params=cfg['model'],
    data_params=cfg['data'],
    training_params=cfg['training'],
    log_dir=EXP_DIR
)
if hasattr(model, "_build_models"):
    try:
        model._build_models()
    except Exception:
        pass

target = getattr(model, "autoencoder", model)
target.load_weights(str(CKPT_PREFIX)).expect_partial()
print("Loaded weights from:", CKPT_PREFIX)

# === 遍历缓存 npz（按标签对齐的 ID 列表） ===
with open(ID_LIST, "r") as f:
    id_list = [ln.strip() for ln in f if ln.strip()]
npz_files = [CACHE_DIR / f"{vid}.npz" for vid in id_list]

print(f"Found {len(npz_files)} labeled videos to encode")

bad_list = OUT_DIR.parent / "bad_npz_list.txt"

for i, f in enumerate(npz_files, 1):
    out_path = OUT_DIR / (f.stem + ".npy")

    # 断点续跑：已存在就跳过
    if out_path.exists():
        print(f"[{i:05d}/{len(npz_files)}] skip (exists) {out_path.name}")
        continue

    # ---------- 读取并“真正解压”到内存（在同一 try 内） ----------
    try:
        with np.load(f) as d:
            key = None
            for k in ("frames", "video", "imgs", "images"):
                if k in d.files:
                    key = k
                    break
            if key is None:
                raise KeyError(f"no frames key in {list(d.files)}")

            # 关键：强制将 npz 内的数组完整解压成内存拷贝
            frames = np.array(d[key], dtype=np.float32, copy=True)

    except Exception as e:
        msg = f"[{i:05d}/{len(npz_files)}] [skip bad: read] {f.name} -> {e}"
        print(msg)
        try:
            with open(bad_list, "a") as bf:
                bf.write(msg + "\n")
        except Exception:
            pass
        continue

    # ---------- 形状/像素域标准化 ----------
    try:
        # (T,H,W) -> (T,H,W,1)；彩色 -> 灰度
        if frames.ndim == 3:  # (T,H,W)
            frames = frames[:, :, :, None]
        elif frames.ndim == 4 and frames.shape[-1] == 3:
            frames = 0.299*frames[...,0:1] + 0.587*frames[...,1:2] + 0.114*frames[...,2:3]
        assert frames.ndim == 4 and frames.shape[-1] == 1, f"unexpected shape {frames.shape}"

        # 缓存有时为 [0,255]；统一到 [0,1]
        if float(frames.max()) > 1.5:
            frames = frames / 255.0

        # 与训练一致的“按视频标准化”
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)

    except Exception as e:
        msg = f"[{i:05d}/{len(npz_files)}] [skip bad: preprocess] {f.name} -> {e}"
        print(msg)
        try:
            with open(bad_list, "a") as bf:
                bf.write(msg + "\n")
        except Exception:
            pass
        continue

    # ---------- 批处理编码（自适应 batch，遇 OOM 自动减半） ----------
    try:
        batch = 256
        latents = []
        s = 0
        while s < len(frames):
            cur = min(batch, len(frames) - s)
            try:
                x = tf.convert_to_tensor(frames[s:s+cur])
                y = target(x, training=False)
                code = y[0] if isinstance(y, (tuple, list)) else y
                latents.append(code.numpy())
                s += cur
            except tf.errors.ResourceExhaustedError:
                if batch <= 8:
                    raise
                batch //= 2
                print(f"[warn] OOM, reduce batch -> {batch}")

        latents = np.concatenate(latents, axis=0)  # (T,128)

        np.save(out_path, latents)
        print(f"[{i:05d}/{len(npz_files)}] saved {out_path.name}  shape={latents.shape}")

    except Exception as e:
        msg = f"[{i:05d}/{len(npz_files)}] [skip bad: encode] {f.name} -> {e}"
        print(msg)
        try:
            with open(bad_list, "a") as bf:
                bf.write(msg + "\n")
        except Exception:
            pass
        continue

print("Done. Latents saved to:", OUT_DIR)
print("Bad npz list (if any):", bad_list)

