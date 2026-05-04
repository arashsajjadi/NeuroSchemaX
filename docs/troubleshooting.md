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
- Confirm `onnx` is installed: `pip install onnx`.
- Some very new ONNX opsets may not be recognised; use a manual spec dict instead.

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
