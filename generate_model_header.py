"""
generate_model_header.py
Export CNN weights (BatchNorm folded into Conv) as a self-contained C++ header.
The header includes the weights AND the cnn_infer() forward pass — no TFLite needed.
"""
from __future__ import annotations

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = "models/cnn_model_best.h5"
NORM_PATH  = "models/cnn_norm_stats.json"
OUTPUT_H   = "firmware/include/model_weights.h"


def fold_bn(W, bias, gamma, beta, mean, var, eps=1e-3):
    """Fold BatchNorm parameters into preceding Conv2D weights and biases."""
    scale  = gamma / np.sqrt(var + eps)
    W_fold = W * scale[np.newaxis, np.newaxis, np.newaxis, :]
    b_fold = (bias - mean) * scale + beta
    return W_fold.astype(np.float32), b_fold.astype(np.float32)


def floats_to_c(arr: np.ndarray, name: str) -> str:
    flat = arr.flatten()
    rows = []
    for i in range(0, len(flat), 6):
        rows.append("    " + ", ".join(f"{v:.8f}f" for v in flat[i:i+6]))
    return f"static const float {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"


FWD_PASS = r"""
// ─── Forward pass helpers ─────────────────────────────────────────────────────
// All activations are in PSRAM (buf_a / buf_b passed by caller).
// Weights are const float[] in flash (.rodata), read via cache.

static void _conv2d_relu(
    const float* in,  int H, int W, int IC,
    float*       out, int OC,
    const float* weight, int KH, int KW,
    const float* bias)
{
    int pH = KH / 2, pW = KW / 2;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < OC; oc++) {
                float sum = bias[oc];
                for (int kh = 0; kh < KH; kh++) {
                    int ih = h + kh - pH;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw2 = 0; kw2 < KW; kw2++) {
                        int iw = w + kw2 - pW;
                        if (iw < 0 || iw >= W) continue;
                        for (int ic = 0; ic < IC; ic++) {
                            sum += in[(ih * W + iw) * IC + ic]
                                 * weight[((kh * KW + kw2) * IC + ic) * OC + oc];
                        }
                    }
                }
                out[(h * W + w) * OC + oc] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
}

static void _maxpool2x2(const float* in, int H, int W, int C, float* out)
{
    int OH = H / 2, OW = W / 2;
    for (int h = 0; h < OH; h++) {
        for (int w = 0; w < OW; w++) {
            for (int c = 0; c < C; c++) {
                float m = -1e38f;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        float v = in[((h * 2 + kh) * W + (w * 2 + kw)) * C + c];
                        if (v > m) m = v;
                    }
                }
                out[(h * OW + w) * C + c] = m;
            }
        }
    }
}

static void _global_avg_pool(const float* in, int H, int W, int C, float* out)
{
    float inv = 1.0f / (float)(H * W);
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                sum += in[(h * W + w) * C + c];
        out[c] = sum * inv;
    }
}

static void _dense_relu(const float* in, int IN, float* out, int OUT,
                        const float* w, const float* b)
{
    for (int o = 0; o < OUT; o++) {
        float sum = b[o];
        for (int i = 0; i < IN; i++) sum += in[i] * w[i * OUT + o];
        out[o] = sum > 0.0f ? sum : 0.0f;
    }
}

static void _dense_softmax(const float* in, int IN, float* out, int OUT,
                            const float* w, const float* b)
{
    for (int o = 0; o < OUT; o++) {
        float sum = b[o];
        for (int i = 0; i < IN; i++) sum += in[i] * w[i * OUT + o];
        out[o] = sum;
    }
    float maxv = out[0];
    for (int o = 1; o < OUT; o++) if (out[o] > maxv) maxv = out[o];
    float denom = 0.0f;
    for (int o = 0; o < OUT; o++) { out[o] = expf(out[o] - maxv); denom += out[o]; }
    float inv = 1.0f / denom;
    for (int o = 0; o < OUT; o++) out[o] *= inv;
}

// ─── Entry point ──────────────────────────────────────────────────────────────
//
//  mel    : flat float32 [97 * 64], z-score normalised (same as training)
//  buf_a  : PSRAM scratch ≥ 200,000 floats (≈ 775 KB)
//  buf_b  : PSRAM scratch ≥  50,000 floats (≈ 192 KB)
//  probs  : output [4] — fall, cough, normal, other
//
static void cnn_infer(const float* mel, float* buf_a, float* buf_b, float probs[4])
{
    // Block 1: Conv(32,3x3)+ReLU [97,64,1]->[97,64,32], MaxPool->[48,32,32]
    _conv2d_relu(mel,   97, 64,  1, buf_a, 32, conv0_w, 3, 3, conv0_b);
    _maxpool2x2 (buf_a, 97, 64, 32, buf_b);

    // Block 2: Conv(64,3x3)+ReLU [48,32,32]->[48,32,64], MaxPool->[24,16,64]
    _conv2d_relu(buf_b, 48, 32, 32, buf_a, 64, conv1_w, 3, 3, conv1_b);
    _maxpool2x2 (buf_a, 48, 32, 64, buf_b);

    // Block 3: Conv(128,3x3)+ReLU [24,16,64]->[24,16,128], MaxPool->[12,8,128]
    _conv2d_relu(buf_b, 24, 16, 64, buf_a, 128, conv2_w, 3, 3, conv2_b);
    _maxpool2x2 (buf_a, 24, 16, 128, buf_b);

    // Head: GAP -> Dense(64)+ReLU -> Dense(4)+Softmax
    float gap[128], d1[64];
    _global_avg_pool(buf_b, 12,  8, 128, gap);
    _dense_relu     (gap,  128, d1,  64, dense0_w, dense0_b);
    _dense_softmax  (d1,    64, probs, 4, dense1_w, dense1_b);
}
"""


def main() -> None:
    print(f"Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    layers = model.layers
    conv_layers  = [(i, l) for i, l in enumerate(layers) if isinstance(l, tf.keras.layers.Conv2D)]
    bn_layers    = [(i, l) for i, l in enumerate(layers) if isinstance(l, tf.keras.layers.BatchNormalization)]
    dense_layers = [l for l in layers if isinstance(l, tf.keras.layers.Dense)]

    lines: list[str] = [
        "// Auto-generated by generate_model_header.py — do not edit\n",
        "#pragma once\n",
        "#include <math.h>\n\n",
    ]

    for conv_idx, (ci, conv) in enumerate(conv_layers):
        W = conv.kernel.numpy()          # (kH, kW, IC, OC)
        b = (conv.bias.numpy()
             if conv.use_bias
             else np.zeros(W.shape[3], dtype=np.float32))

        # Find the first BN layer that comes after this conv
        bn = next((l for bi, l in bn_layers if bi > ci), None)
        if bn is not None:
            gamma, beta, mean, var = [w.numpy() for w in bn.weights]
            W, b = fold_bn(W, b, gamma, beta, mean, var)

        kH, kW, IC, OC = W.shape
        lines.append(f"// Conv{conv_idx}: kernel ({kH},{kW},{IC})->{OC}, BN folded\n")
        lines.append(floats_to_c(W, f"conv{conv_idx}_w"))
        lines.append(floats_to_c(b, f"conv{conv_idx}_b"))
        lines.append("\n")
        print(f"  Conv{conv_idx}: {W.shape}  bias {b.shape}")

    for di, dense in enumerate(dense_layers):
        W = dense.kernel.numpy()   # (IN, OUT)
        b = dense.bias.numpy()
        IN, OUT = W.shape
        lines.append(f"// Dense{di}: {IN}->{OUT}\n")
        lines.append(floats_to_c(W, f"dense{di}_w"))
        lines.append(floats_to_c(b, f"dense{di}_b"))
        lines.append("\n")
        print(f"  Dense{di}: {W.shape}  bias {b.shape}")

    lines.append(FWD_PASS)

    Path(OUTPUT_H).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_H).write_text("".join(lines), encoding="utf-8")
    size_kb = Path(OUTPUT_H).stat().st_size // 1024
    print(f"\nWrote {OUTPUT_H}  ({size_kb} KB)")
    print("Next: rebuild firmware (TFLite removed — no lib_deps needed)")


if __name__ == "__main__":
    main()
