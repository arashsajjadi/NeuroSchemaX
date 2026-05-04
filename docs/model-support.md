# Model Support

## Supported inputs

| Source | Notes |
|---|---|
| `.onnx` file path | Standard ONNX format |
| `.json` file path | Manual spec JSON |
| `.yaml` / `.yml` file path | Manual spec YAML |
| Python `dict` | Manual spec as a dict |
| `torch.nn.Module` | Requires `pip install torch` |
| `tf.keras.Model` | Requires `pip install tensorflow` |
| `onnx.ModelProto` | Pre-loaded ONNX object |

## Supported outputs

| Format | Function / method | Notes |
|---|---|---|
| HTML | `save_network_html` | Self-contained, no server needed |
| SVG | `save_network_svg` | Requires Playwright + Chromium |
| Paper JSON | `save_paper_json` | Compact, publication-ready |
| Debug JSON | `save_debug_json` | Full layer attributes |
| NN-SVG JSON | `save_nnsvg_spec` | Loadable in NN-SVG web app |

## NN-SVG rendering families

| Family | When used | Layers supported |
|---|---|---|
| `fcnn` | Pure dense / MLP | Dense, activation |
| `lenet` | Small CNNs (1-3 conv layers) | Conv, pool, dense |
| `alexnet` | Deeper CNNs (4+ conv layers) | Conv, pool, dense |

## Architecture detection

The recommended family is chosen automatically based on:

1. Number of convolutional layers
2. Presence of skip connections
3. Presence of attention / recurrent layers

See `recommend_view()` for the full reasoning including a `reason` string.

## Known limitations

- Skip / residual connections are **not drawn**; in summary mode each
  Residual Block is labeled `+skip collapsed` so it is visible that links
  were dropped.  Full graph stays in `export-debug-json`.
- Attention (Transformer) and recurrent (LSTM/GRU) layers are rendered as a
  **block-level rectangle sequence** (`transformer_mode="block_summary"`),
  not as exact attention/Q/K/V graphs.  Use `transformer_mode="unsupported"`
  to opt into a diagnostic page instead of an approximate summary.
- Encoder-decoder (U-Net) architectures are rendered as
  `Encoder → Bottleneck → Decoder → Segmentation Head` blocks with
  `concat collapsed` markers — concat / upsample edges are not drawn.
- Branching / multi-path topologies collapse to a single sequential path.

See [limitations.md](limitations.md) for the full honesty table.
