from __future__ import annotations
import torch
from torch import nn
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from pathlib import Path
import json

def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError('Unsupported model: could not locate transformer layers')

def get_hidden_size(model: nn.Module) -> int:
    config = getattr(model, 'config', None)
    for attr in ('hidden_size', 'n_embd', 'd_model'):
        if config is not None and hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError('Could not infer model hidden size')

def _extract_hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output:
        return output[0]
    raise ValueError('Hook output does not contain a hidden-state tensor')

def _replace_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple) and output:
        return (hidden, *output[1:])
    raise ValueError('Hook output does not contain a hidden-state tensor')

class SelectiveLayerCapture:
    """Capture block outputs at specific layers using forward hooks.

    Usage:
        capture = SelectiveLayerCapture(model, target_layers=[22, 23, 24])
        with capture:
            outputs = model(**inputs, use_cache=False)
        acts = capture.extract_positions(positions_per_item, batch_size)
        capture.clear()
    """

    def __init__(self, model: nn.Module, target_layers: Optional[List[int]]=None, save_dtype: torch.dtype=torch.float16, pin_memory: bool=True):
        self.model = model
        self.layers = get_transformer_layers(model)
        self.n_layers = len(self.layers)
        self.target_layers = target_layers if target_layers is not None else list(range(self.n_layers))
        self.save_dtype = save_dtype
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self._captured: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, layer_idx: int):

        def hook(module, input, output):
            hidden = _extract_hidden(output)
            t = hidden.detach().to(dtype=self.save_dtype, device='cpu', non_blocking=True)
            if self.pin_memory:
                t = t.pin_memory()
            self._captured[layer_idx] = t
        return hook

    def __enter__(self):
        for layer_idx in self.target_layers:
            h = self.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def extract_positions(self, positions_per_item: List[Dict[str, Optional[int]]], batch_size: int, position_names: Sequence[str]=('last_token', 'instruction_last', 'endorsement_last')) -> Dict[str, torch.Tensor]:
        """Extract activations at specific positions for each batch item.

        Returns:
            {position_name: Tensor(batch, n_target_layers, hidden_dim)}
        """
        h_dim = next(iter(self._captured.values())).shape[-1]
        n_target = len(self.target_layers)
        result: Dict[str, torch.Tensor] = {}
        for pos_name in position_names:
            buf = torch.zeros(batch_size, n_target, h_dim, dtype=self.save_dtype)
            for item_idx in range(batch_size):
                pos = positions_per_item[item_idx].get(pos_name)
                if pos is None:
                    continue
                for li, layer_idx in enumerate(self.target_layers):
                    buf[item_idx, li] = self._captured[layer_idx][item_idx, pos, :]
            result[pos_name] = buf
        return result

    def extract_all_positions_flat(self, positions_per_item: List[Dict[str, Optional[int]]], batch_size: int, position_names: Sequence[str]=('last_token', 'instruction_last', 'endorsement_last')) -> Dict[str, torch.Tensor]:
        """Like extract_positions but returns (batch, n_all_layers, hidden_dim)
        where n_all_layers = n_model_layers + 1 (for backward compat with hidden_states indexing).

        Non-target layers are filled with zeros.
        """
        h_dim = next(iter(self._captured.values())).shape[-1]
        n_all = self.n_layers + 1
        result: Dict[str, torch.Tensor] = {}
        for pos_name in position_names:
            buf = torch.zeros(batch_size, n_all, h_dim, dtype=self.save_dtype)
            for item_idx in range(batch_size):
                pos = positions_per_item[item_idx].get(pos_name)
                if pos is None:
                    continue
                for layer_idx in self.target_layers:
                    hs_idx = layer_idx + 1
                    buf[item_idx, hs_idx] = self._captured[layer_idx][item_idx, pos, :]
            result[pos_name] = buf
        return result

    def clear(self):
        self._captured.clear()
