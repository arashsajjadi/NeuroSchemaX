# Troubleshooting

## Run the doctor

```bash
neuroschemax doctor
```

This prints a pass/fail summary.  Address any action items listed.

---

## SVG export fails with `BrowserNotAvailableError`

SVG export requires Playwright and the Chromium browser.

```bash
pip install playwright
playwright install chromium
```

Then retry.

---

## Missing assets

If `doctor` reports missing JS assets:

```bash
pip install --force-reinstall neuroschemax
```

---

## ONNX file does not parse

- Confirm the file is a valid `.onnx` file (not corrupted).
- Confirm `onnx` is installed: `pip install onnx` (or `pip install neuroschemax[onnx]`).
- Re-export with a lower opset: `torch.onnx.export(..., opset_version=17)`.
- Some very new ONNX opsets may not be recognised; use a manual spec dict instead.

## PyTorch → ONNX export fails with `ModuleNotFoundError: No module named 'onnxscript'`

`torch >= 2.x` requires `onnxscript` for the new ONNX exporter path.

```bash
pip install onnxscript
```

Then retry.  If `onnxscript` is not available, fall back to the legacy exporter:

```python
torch.onnx.export(model, dummy, "model.onnx", opset_version=16)
```

## Diagram shapes show `?`

Shape propagation is best-effort.  Common causes:

- **Dynamic batch axes** — use a concrete batch size at export time.
- **ONNX models without shape info** — run `onnx.shape_inference.infer_shapes`
  before passing to NeuroSchemaX (the adapter does this automatically).
- **BatchNorm/Dropout removed by ONNX exporter in eval mode** — these layers
  are folded out during export and will not appear in the diagram.  This is
  expected; the diagram reflects the exported computation graph.

## Inline Colab display is broken or shows raw JS

Colab restricts inline HTML/JS rendering.  The recommended approach:

```python
fig.save_html("diagram.html")
# Download via Files panel → right-click → Download
# Open in Chrome or Firefox for full interactivity.
```

Install `neuroschemax[colab]` for `IPython` support (`pip install "neuroschemax[colab]"`).
The IFrame preview in Colab is limited; the downloaded HTML file always works.

---

## YAML file does not parse

- Confirm `PyYAML` is installed: `pip install PyYAML`.
- Confirm the YAML is valid (no tab characters, correct indentation).

---

## HTML diagram is blank

- Open the HTML file in a modern browser (Chrome, Firefox, Edge).
- Check the browser console for JavaScript errors.
- Try regenerating with `--theme debug` to see extra diagnostics in the output.

---

## ImportError on `neuroschemax`

Reinstall in editable mode from the project root:

```bash
pip install -e .
```

---

## Confidence is LOW or UNKNOWN

This means the architecture did not map cleanly to an NN-SVG family.
The diagram is still generated as a best-effort FCNN approximation.
Check `warnings` in the `recommend-view` output for details.
