"""Fuse consecutive layers into logical blocks / stages."""

from __future__ import annotations

from ..core.enums import LayerKind
from ..ir.semantic_ir import SemanticBlock, SemanticLayer

# Layer kinds that typically appear *inside* a block but don't start one
_AUXILIARY_KINDS = frozenset({
    LayerKind.BATCH_NORM, LayerKind.LAYER_NORM, LayerKind.GROUP_NORM,
    LayerKind.INSTANCE_NORM, LayerKind.ACTIVATION, LayerKind.RELU,
    LayerKind.SIGMOID, LayerKind.TANH, LayerKind.GELU,
    LayerKind.DROPOUT, LayerKind.PAD,
})

# Layer kinds that signal the start of a new "primary" computation
_PRIMARY_KINDS = frozenset({
    LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV,
    LayerKind.DENSE,
    LayerKind.POOL_MAX, LayerKind.POOL_AVG, LayerKind.POOL_GLOBAL,
    LayerKind.ATTENTION, LayerKind.LSTM, LayerKind.GRU, LayerKind.RECURRENT,
})


def fuse_blocks(layers: list[SemanticLayer]) -> list[SemanticBlock]:
    """Group a flat layer list into sequential blocks.

    Heuristic: every *primary* layer starts a new block; subsequent
    auxiliary layers attach to it.
    """
    blocks: list[SemanticBlock] = []
    current_ids: list[str] = []
    block_idx = 0

    for layer in layers:
        if layer.kind in (LayerKind.INPUT, LayerKind.OUTPUT):
            continue
        if layer.kind in _PRIMARY_KINDS and current_ids:
            blocks.append(SemanticBlock(
                id=f"block_{block_idx}",
                name=f"block_{block_idx}",
                layer_ids=current_ids,
            ))
            block_idx += 1
            current_ids = []
        current_ids.append(layer.id)

    if current_ids:
        blocks.append(SemanticBlock(
            id=f"block_{block_idx}",
            name=f"block_{block_idx}",
            layer_ids=current_ids,
        ))

    return blocks
