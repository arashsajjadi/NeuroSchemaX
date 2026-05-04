# Limitations

NeuroSchemaX is a practical tool with clearly defined bounds.  Understanding
them helps you interpret results correctly and know when to rely on the
debug JSON instead of the visual diagram.

## NN-SVG renders sequential layouts only

NN-SVG draws architectures in three families:

- **FCNN** — columns of fully-connected neurons (classic MLP).
- **LeNet** — sequence of conv / pool / dense stages (small CNN).
- **AlexNet** — deeper conv stack with an optional dense tail.

All three families are strictly sequential: layers go left to right, one
column per layer.  NN-SVG does not have native support for:

- skip / residual connections drawn as curved edges
- multi-branch inputs (Inception, detection heads)
- encoder-decoder paths (U-Net)
- recurrent loops
- attention blocks rendered as graphs
- arbitrary custom DAG operations

NeuroSchemaX does **not** fake support for these structures.  It renders the
nearest sequential approximation honestly, with explicit on-diagram markers
and warnings.

## Three honest output levels

| Level | When | What you see |
|---|---|---|
| **1. Exact** | MLPs, small CNNs, sequential VGG/AlexNet-style CNNs | NN-SVG diagram with operation-aware labels |
| **2. Architecture-aware summary** | Large CNNs, ResNet-like, U-Net-like, Transformer-like | Labeled rectangular blocks with `+skip collapsed`, `concat collapsed`, `[MH-Attn]`, `[FFN]` markers |
| **3. Diagnostic page** | `transformer_mode="unsupported"` | A wide labeled block explaining what was detected and directing the user to the debug JSON or to `block_summary` |

Level 2 is **approximate by construction** — the diagram itself signals
that skip / concat / attention internals are not drawn.  Level 3 is the
explicit "this cannot be rendered exactly" page.

## How approximate rendering works

When a model cannot be represented exactly, NeuroSchemaX:

1. **Picks the closest NN-SVG family** for the sequential backbone.
2. **Skips structural ops** (Add, Concat, Multiply, Upsample) — they are not
   neuron columns and should not appear as such.
3. **Surfaces supporting ops as inline badges** (`+BN`, `+LN`, `+Drop 0.5`)
   so they remain visible without crowding the diagram.
4. **Shows an amber warning badge** in the HTML explaining exactly what was
   approximated.
5. **Reports reduced confidence** — `recommend_view()` returns
   `"confidence": "medium"` or `"low"` and `"is_approximate": true`.
6. **Preserves the full graph** in the debug-JSON export.

The approximate diagram is still readable and useful for understanding layer
types, channels, and the general structure.  It just does not show skip
connections or branches as curved edges.

## Fidelity table

| Model type | Rendered as | Visual fidelity | Where the rest is |
|---|---|---|---|
| MLP / dense network | FCNN | **Exact** | — |
| Small CNN (≤ 3 convs) | LeNet | **Exact** | — |
| VGG-style deep CNN | AlexNet | **Exact** | — |
| ResNet / residual blocks | AlexNet backbone *or* `Stem → Residual Block N (+skip collapsed) → Head` summary | **Approximate** | Skip links in debug JSON |
| U-Net / encoder-decoder | AlexNet backbone *or* `Encoder → Bottleneck → Decoder → Segmentation Head` summary | **Approximate** | Decoder branches + concat in debug JSON |
| Transformer / attention | Block-level rectangle summary (`block_summary`) or diagnostic page (`unsupported`) | **Approximate** | All layers + attention attributes in debug JSON |
| LSTM / GRU / RNN | Block-level sequence | **Approximate** | All layers in debug JSON |
| Object-detection head | AlexNet backbone | **Approximate** | Detection branches in debug JSON |
| Unknown ops | Generic nodes / skipped | **Low** | Raw op types in debug JSON |

We do **not** claim:
- exact Transformer attention flow
- exact ResNet skip-edge drawing
- exact U-Net encoder-decoder skip topology
- arbitrary DAG rendering

## Debug JSON preserves what the diagram cannot show

Running `export-debug-json` (or calling `save_debug_json()`) produces a
verbose JSON file that includes:

- all layers with their raw op types and attributes (heads, d_model, dropout
  rate, kernel/stride/padding, …)
- skip connections and merge nodes detected in the graph
- shape information for every tensor where available
- confidence scores per layer
- warnings about approximations
- complexity classification (`sequential`, `skip`, `multi_branch`, `dag`)

This is the authoritative record of the model structure for complex
architectures.

## Optional-dependency limitations

- **SVG export** requires Playwright and Chromium.  HTML export works without
  them; if you try SVG without Playwright, you get a clear error with
  install instructions.
- **PyTorch adapter** walks the module tree but does not capture edge
  connections between layers.  For richer graphs, export to ONNX first.
- **TensorFlow adapter** walks the Keras layer list.  Custom layers may
  appear as `unknown`.

## Shape inference

Shape propagation is best-effort.  Models with dynamic axes or symbolic
dimensions fall back to `?` placeholders.  Labels may show `?` where a
concrete dimension could not be determined.  The diagram is still generated;
only the shape annotation is missing.

## Confidence levels

Every recommendation carries a `confidence` level:

| Confidence | Meaning |
|---|---|
| `high` | Architecture maps cleanly to an NN-SVG family |
| `medium` | Mostly clean, but some features (skip connections) are approximated |
| `low` | Significant features (attention, recurrent, DAG) cannot be rendered |
| `unknown` | No recognisable standard layers found |

Check `recommend_view()` whenever a diagram looks unexpected.  The `reason`
and `warnings` fields explain the decision.
