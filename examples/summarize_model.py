"""Print a text / Markdown summary of a model."""

from __future__ import annotations

import neuroschemax as nsx


SPEC = {
    "model_name": "resnet_like",
    "layers": [
        {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
        {"name": "conv1", "kind": "conv", "out_channels": 64, "kernel_size": [7, 7], "stride": [2, 2]},
        {"name": "bn1", "kind": "batchnormalization"},
        {"name": "relu1", "kind": "relu"},
        {"name": "pool1", "kind": "maxpool", "kernel_size": [3, 3], "stride": [2, 2]},
        {"name": "conv2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
        {"name": "bn2", "kind": "batchnormalization"},
        {"name": "relu2", "kind": "relu"},
        {"name": "conv3", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
        {"name": "gap", "kind": "globalaveragepool"},
        {"name": "fc", "kind": "dense", "units": 1000},
    ],
}


def main() -> None:
    arch = nsx.parse_model(SPEC)
    print("=" * 60)
    print("TEXT SUMMARY")
    print("=" * 60)
    print(nsx.summarize_model(arch))
    print()
    print("=" * 60)
    print("MARKDOWN SUMMARY")
    print("=" * 60)
    print(nsx.summarize_model(arch, format="markdown"))


if __name__ == "__main__":
    main()
