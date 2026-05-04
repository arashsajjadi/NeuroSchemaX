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
- arbitrary custom operations

NeuroSchemaX does **not** fake support for these structures.  It renders the
nearest sequential approximation honestly, with explicit warnings.

## How approximate rendering works

When a model cannot be represented exactly, NeuroSchemaX:

1. **Picks the closest NN-SVG family** for the sequential backbone.
2. **Skips structural ops** (Add, Concat, Multiply, Upsample) — they are not
   neuron columns and should not appear as such.
3. **Shows an amber warning badge** in the HTML explaining exactly what was
   approximated.
4. **Reports reduced confidence** — `recommend_view()` returns
   `"confidence": "medium"` or `"low"` and `"is_approximate": true`.
5. **Preserves the full graph** in the debug-JSON export.

The approximate diagram is still readable and useful for understanding layer
types, channels, and the general structure.  It just does not show skip
connections or branches.

## Fidelity table

| Model type | Rendered as | Visual fidelity | Where the rest is |
|---|---|---|---|
| MLP / dense network | FCNN | High | — |
| Small CNN (≤ 3 convs) | LeNet | High | — |
| VGG-style deep CNN | AlexNet | High | — |
| ResNet / residual blocks | AlexNet backbone | Approximate | Skip links in debug JSON |
| U-Net / encoder-decoder | AlexNet backbone | Approximate | Decoder branches in debug JSON |
| Transformer / attention | FCNN nodes | Approximate | Attention blocks in debug JSON |
| LSTM / GRU / RNN | FCNN nodes | Approximate | Recurrent structure in debug JSON |
| Object-detection head | AlexNet backbone | Approximate | Detection branches in debug JSON |
| Unknown ops | Shown as generic nodes | Low | Raw op types in debug JSON |

## Debug JSON preserves what the diagram cannot show

Running `export-debug-json` (or calling `save_debug_json()`) produces a
verbose JSON file that includes:

- all layers with their raw op types and attributes
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
