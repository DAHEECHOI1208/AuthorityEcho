"""AB upstream excerpts, unchanged bodies. MIT notice: licenses/AuthorityBias_LICENSE.txt.
Commit f690794951f50542929ab7093db9f4719f9dd7b9. See M00_upstream_excerpts.json.
Imports assembled locally; only these functions/classes are vendored."""
from __future__ import annotations
import torch
from torch import nn
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from pathlib import Path
import json
from dataclasses import dataclass
from mechanism.native.C14_ab_capture import _extract_hidden, _replace_hidden

def normalize_torch(vector: torch.Tensor) -> torch.Tensor:
    vec = vector.float()
    norm = vec.norm()
    if float(norm) == 0.0:
        return vec
    return vec / norm

def project_out_torch(vector: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None:
        return vector.float()
    result = vector.float()
    basis_rows = basis.float()
    if basis_rows.ndim == 1:
        basis_rows = basis_rows.unsqueeze(0)
    for direction in basis_rows:
        denom = torch.dot(direction, direction)
        if float(denom) == 0.0:
            continue
        result = result - torch.dot(result, direction) / denom * direction
    return result

@dataclass
class TargetSpec:
    layer: int
    component: str
    direction: Optional[torch.Tensor]
    patch_mean: Optional[torch.Tensor]
    nuisance_basis: Optional[torch.Tensor] = None

def _scale_delta(base_hidden: torch.Tensor, direction: torch.Tensor, alpha: float, scaling: str) -> torch.Tensor:
    if scaling == 'none':
        return alpha * direction
    if scaling == 'resid_norm':
        scale = float(base_hidden.norm().item())
        return alpha * scale * direction
    if scaling == 'resid_std':
        scale = float(base_hidden.std(unbiased=False).item())
        return alpha * scale * direction
    raise ValueError(f'Unknown scaling mode: {scaling}')

def _make_intervention_hook(*, spec: TargetSpec, alpha: float, intervention_mode: str, norm_scaling: str, apply_prompt: bool, apply_decode: bool, prompt_positions_by_row: Sequence[Sequence[int]]):

    def hook(_module, _inputs, output):
        hidden = _extract_hidden(output)
        is_decode_step = hidden.shape[1] == 1
        if is_decode_step:
            if not apply_decode:
                return output
            positions_by_row = [[0] for _ in range(hidden.shape[0])]
        else:
            if not apply_prompt:
                return output
            positions_by_row = prompt_positions_by_row
        updated = hidden.clone()
        d = None
        p = None
        if spec.direction is not None:
            d = spec.direction.to(hidden.device, dtype=hidden.dtype)
        if spec.patch_mean is not None:
            p = spec.patch_mean.to(hidden.device, dtype=hidden.dtype)
        nuisance_basis = None
        if spec.nuisance_basis is not None:
            nuisance_basis = spec.nuisance_basis.to(hidden.device, dtype=hidden.dtype)
        for row_idx, row_positions in enumerate(positions_by_row):
            if row_idx >= hidden.shape[0]:
                break
            valid_positions = [int(pos) for pos in row_positions if 0 <= int(pos) < hidden.shape[1]]
            if not valid_positions:
                continue
            for pos in valid_positions:
                base = hidden[row_idx, pos, :]
                if intervention_mode in {'add', 'subtract'}:
                    if d is None:
                        continue
                    direction = d if nuisance_basis is None else normalize_torch(project_out_torch(d, nuisance_basis))
                    if float(direction.norm()) == 0.0:
                        continue
                    delta = _scale_delta(base, direction, alpha, norm_scaling)
                    if intervention_mode == 'subtract':
                        delta = -delta
                    updated[row_idx, pos, :] = base + delta
                elif intervention_mode == 'replace_mean':
                    if p is None:
                        continue
                    target = p
                    if nuisance_basis is not None:
                        target = base + project_out_torch(p - base, nuisance_basis)
                    updated[row_idx, pos, :] = target
                elif intervention_mode == 'interpolate_mean':
                    if p is None:
                        continue
                    delta_to_patch = p - base
                    if nuisance_basis is not None:
                        delta_to_patch = project_out_torch(delta_to_patch, nuisance_basis)
                    updated[row_idx, pos, :] = base + alpha * delta_to_patch
                elif intervention_mode == 'project_out_direction':
                    if d is None:
                        continue
                    direction = d if nuisance_basis is None else normalize_torch(project_out_torch(d, nuisance_basis))
                    if float(direction.norm()) == 0.0:
                        continue
                    coeff = torch.dot(base, direction)
                    updated[row_idx, pos, :] = base - alpha * coeff * direction
                elif intervention_mode == 'project_out_subspace':
                    if nuisance_basis is None:
                        raise ValueError('--intervention-mode project_out_subspace requires --project-out-subspace-path')
                    residual = project_out_torch(base, nuisance_basis)
                    projection = base - residual
                    updated[row_idx, pos, :] = base - alpha * projection
                else:
                    raise ValueError(f'Unknown intervention mode: {intervention_mode}')
        return _replace_hidden(output, updated)
    return hook

def _make_intervention_hook_explicit_phase(*, phase: str, spec: TargetSpec, alpha: float, intervention_mode: str, norm_scaling: str, apply_prompt: bool, apply_decode: bool, prompt_positions_by_row: Sequence[Sequence[int]]):

    def hook(_module, _inputs, output):
        hidden = _extract_hidden(output)
        is_decode_step = phase == 'decode'
        if is_decode_step:
            if not apply_decode:
                return output
            positions_by_row = [[0] for _ in range(hidden.shape[0])]
        else:
            if not apply_prompt:
                return output
            positions_by_row = prompt_positions_by_row
        updated = hidden.clone()
        d = None
        p = None
        if spec.direction is not None:
            d = spec.direction.to(hidden.device, dtype=hidden.dtype)
        if spec.patch_mean is not None:
            p = spec.patch_mean.to(hidden.device, dtype=hidden.dtype)
        nuisance_basis = None
        if spec.nuisance_basis is not None:
            nuisance_basis = spec.nuisance_basis.to(hidden.device, dtype=hidden.dtype)
        for row_idx, row_positions in enumerate(positions_by_row):
            if row_idx >= hidden.shape[0]:
                break
            valid_positions = [int(pos) for pos in row_positions if 0 <= int(pos) < hidden.shape[1]]
            if not valid_positions:
                continue
            for pos in valid_positions:
                base = hidden[row_idx, pos, :]
                if intervention_mode in {'add', 'subtract'}:
                    if d is None:
                        continue
                    direction = d if nuisance_basis is None else normalize_torch(project_out_torch(d, nuisance_basis))
                    if float(direction.norm()) == 0.0:
                        continue
                    delta = _scale_delta(base, direction, alpha, norm_scaling)
                    if intervention_mode == 'subtract':
                        delta = -delta
                    updated[row_idx, pos, :] = base + delta
                elif intervention_mode == 'replace_mean':
                    if p is None:
                        continue
                    target = p
                    if nuisance_basis is not None:
                        target = base + project_out_torch(p - base, nuisance_basis)
                    updated[row_idx, pos, :] = target
                elif intervention_mode == 'interpolate_mean':
                    if p is None:
                        continue
                    delta_to_patch = p - base
                    if nuisance_basis is not None:
                        delta_to_patch = project_out_torch(delta_to_patch, nuisance_basis)
                    updated[row_idx, pos, :] = base + alpha * delta_to_patch
                elif intervention_mode == 'project_out_direction':
                    if d is None:
                        continue
                    direction = d if nuisance_basis is None else normalize_torch(project_out_torch(d, nuisance_basis))
                    if float(direction.norm()) == 0.0:
                        continue
                    coeff = torch.dot(base, direction)
                    updated[row_idx, pos, :] = base - alpha * coeff * direction
                elif intervention_mode == 'project_out_subspace':
                    if nuisance_basis is None:
                        raise ValueError('--intervention-mode project_out_subspace requires --project-out-subspace-path')
                    residual = project_out_torch(base, nuisance_basis)
                    projection = base - residual
                    updated[row_idx, pos, :] = base - alpha * projection
                else:
                    raise ValueError(f'Unknown intervention mode: {intervention_mode}')
        return _replace_hidden(output, updated)
    return hook
