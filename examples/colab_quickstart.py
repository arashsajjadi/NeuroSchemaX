"""Google Colab / Jupyter quickstart for NeuroSchemaX.

Run this in a Colab notebook (or Jupyter) cell-by-cell.

Installation (first cell):
    !pip install neuroschemax

Optional — ONNX input for real PyTorch models:
    !pip install "neuroschemax[torch]"
"""

# ── Option 1: Manual architecture spec (no extra dependencies) ──────────────
# This works with the base install; no ONNX/PyTorch needed.

import neuroschemax as nsx

mlp_spec = {
    "model_name": "my_mlp",
    "layers": [
        {"name": "input",  "kind": "input",  "shape": [1, 784]},
        {"name": "fc1",    "kind": "dense",   "units": 256},
        {"name": "relu1",  "kind": "relu"},
        {"name": "fc2",    "kind": "dense",   "units": 128},
        {"name": "relu2",  "kind": "relu"},
        {"name": "output", "kind": "dense",   "units": 10},
    ],
}

fig = nsx.figure(theme="paper")
fig.draw(mlp_spec)

# Inline preview: works in Colab and Jupyter.
# Colab may restrict JavaScript interactivity; the HTML content still renders.
fig.show()
# In Colab you will also see a short note about downloading for full interactivity.

# For the fully interactive diagram, save and download:
fig.save_html("my_mlp.html")
print("Saved: my_mlp.html  →  Files panel → right-click → Download → open in Chrome.")

# _repr_html_ hook: if fig is the last expression in a Jupyter cell,
# the diagram renders inline automatically:
# fig


# ── Option 2: Real PyTorch model → ONNX → NeuroSchemaX ─────────────────────
# Requires:  !pip install "neuroschemax[torch]"

def example_pytorch_onnx() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch not available — skipping PyTorch example.")
        print("Install with:  pip install torch  or  pip install neuroschemax[torch]")
        return

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3)
            self.relu1 = nn.ReLU()
            self.pool  = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.relu2 = nn.ReLU()
            self.fc    = nn.Linear(32 * 5 * 5, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(self.relu1(self.conv1(x)))
            x = self.relu2(self.conv2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = TinyCNN()
    model.eval()

    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, "tiny_cnn.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print("Exported: tiny_cnn.onnx")

    fig2 = nsx.figure(theme="paper")
    fig2.draw("tiny_cnn.onnx")
    fig2.save_html("tiny_cnn.html")
    print("Saved: tiny_cnn.html  →  download and open in Chrome/Firefox.")

    # Show architecture summary in the notebook
    print(nsx.summarize_model("tiny_cnn.onnx"))


# Uncomment to run the PyTorch example:
# example_pytorch_onnx()


# ── Option 3: Transformer block summary (conceptual, not exact) ─────────────

transformer_spec = {
    "model_name": "tiny_transformer",
    "layers": [
        {"name": "embed",  "kind": "embedding"},
        {"name": "pos",    "kind": "add"},
        {"name": "attn1",  "kind": "attention", "num_heads": 8, "d_model": 512},
        {"name": "norm1",  "kind": "layernorm"},
        {"name": "ffn1",   "kind": "dense",     "units": 2048},
        {"name": "gelu",   "kind": "gelu"},
        {"name": "ffn2",   "kind": "dense",     "units": 512},
        {"name": "norm2",  "kind": "layernorm"},
        {"name": "output", "kind": "dense",     "units": 1000},
    ],
}

fig3 = nsx.figure(theme="paper")
fig3.draw(transformer_spec)
fig3.save_html("transformer_block_summary.html")
print("Saved: transformer_block_summary.html")
print("Note: this is a conceptual block summary, not exact Transformer rendering.")
print("Q/K/V, individual heads, and residual paths are not drawn.")
