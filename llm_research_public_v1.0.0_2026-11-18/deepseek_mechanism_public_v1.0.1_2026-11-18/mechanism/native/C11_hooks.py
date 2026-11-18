import numpy as np
from mechanism.native.A04_runtime import TaskFatal

def _extract_hidden(output):
    from mechanism.native.C14_ab_capture import _extract_hidden as upstream
    return upstream(output)

def _replace_hidden(output, hidden):
    from mechanism.native.C14_ab_capture import _replace_hidden as upstream
    return upstream(output, hidden)

def edit_numpy(h, operator, alpha, vector=None, target=None, basis=None):
    """NEW numerical oracle used by CPU tests; same formulas as tensor operations."""
    h = np.asarray(h, dtype=np.float64)
    if not np.isfinite(alpha):
        raise ValueError('Non-finite strength')
    if operator in ('add', 'subtract', 'project_out_direction'):
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != h.shape or not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-12:
            raise ValueError('Invalid/zero direction')
        if operator == 'add':
            return h + alpha * v
        if operator == 'subtract':
            return h - alpha * v
        v = v / np.linalg.norm(v)
        return h - alpha * np.dot(h, v) * v
    if operator in ('interpolate_mean', 'replace_mean', 'patch_activation'):
        t = np.asarray(target, dtype=np.float64)
        if t.shape != h.shape or not np.isfinite(t).all():
            raise ValueError('Invalid target')
        return h + alpha * (t - h) if operator == 'interpolate_mean' else t.copy()
    if operator == 'project_out_subspace':
        b = np.asarray(basis, dtype=np.float64)
        if b.ndim != 2 or b.shape[0] != len(h) or (not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-06)):
            raise ValueError('Basis must be orthonormal columns')
        return h - alpha * b @ (b.T @ h)
    raise ValueError('Unknown operator: ' + operator)

def edit_tensor(h, spec):
    """New validation adapter, executing AB's original operator branches."""
    import torch
    from mechanism.native.J12_ab_operators import TargetSpec, _make_intervention_hook_explicit_phase
    a, op = (float(spec.get('strength', 0)), spec['operator'])
    if not np.isfinite(a) or not torch.isfinite(h).all():
        raise ValueError('Nonfinite edit')

    def tensor(key):
        t = torch.as_tensor(spec[key], device=h.device, dtype=h.dtype)
        if not torch.isfinite(t).all():
            raise ValueError('Nonfinite intervention object')
        return t
    v = t = b = None
    if op in ('add', 'subtract', 'project_out_direction'):
        v = tensor('vector')
        if v.shape != h.shape or v.float().norm().item() <= 1e-12:
            raise ValueError('Invalid direction')
        if op == 'project_out_direction':
            v = v / v.float().norm().to(v.dtype)
    elif op in ('interpolate_mean', 'replace_mean', 'patch_activation'):
        t = tensor('target')
        if t.shape != h.shape:
            raise ValueError('Target shape mismatch')
        if op == 'patch_activation':
            op = 'replace_mean'
    elif op == 'project_out_subspace':
        b = tensor('basis')
        if b.ndim != 2 or b.shape[0] != len(h) or (not torch.allclose(b.float().T @ b.float(), torch.eye(b.shape[1], device=b.device), atol=0.001)):
            raise ValueError('Expected orthonormal column basis')
    else:
        raise ValueError('Unknown operator: ' + op)
    hook = _make_intervention_hook_explicit_phase(phase='prefill', spec=TargetSpec(int(spec.get('layer', 0)), 'block_output', v, t, b.T if b is not None else None), alpha=a, intervention_mode=op, norm_scaling='none', apply_prompt=True, apply_decode=False, prompt_positions_by_row=[[0]])
    return hook(None, None, h.reshape(1, 1, -1))[0, 0]

class HookSession:

    def __init__(self, model, layers, layer_ids, positions, hook_location='block_output', interventions=None, prompt_length=None):
        self.model, self.layers = (model, layers)
        self.layer_ids, self.positions = (layer_ids, positions)
        self.location, self.interventions = (hook_location, interventions or [])
        self.prompt_length = prompt_length
        self.handles, self.captures, self.effects = ([], {}, [])
        self.forward_count = 0
        from mechanism.native.C14_ab_capture import SelectiveLayerCapture
        import torch
        self.upstream_capture = object.__new__(SelectiveLayerCapture)
        self.upstream_capture.save_dtype = torch.float32
        self.upstream_capture.pin_memory = False
        self.upstream_capture._captured = {}
        if self.location not in ('block_input', 'block_output'):
            raise ValueError('hook_location must be block_input or block_output')
        if any((i < 0 or i >= len(layers) for i in layer_ids)):
            raise ValueError('Layer index out of range')

    def root_hook(self, module, args, kwargs):
        self.forward_count += 1

    def process(self, layer_id, hidden):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise TaskFatal('Reference implementation requires batch_size=1')
        phase = 'prefill' if self.forward_count == 1 else 'decode'
        if phase == 'prefill' and hidden.shape[1] != self.prompt_length:
            raise TaskFatal('Prefill length differs from serialized input')
        updated = hidden
        relevant = [s for s in self.interventions if int(s['layer']) == layer_id and s.get('phase', 'prefill') in (phase, 'both')]
        if relevant:
            updated = hidden.clone()
        for spec in relevant:
            indices = [hidden.shape[1] - 1] if phase == 'decode' else spec['token_indices']
            if not indices or any((p < 0 or p >= hidden.shape[1] for p in indices)):
                raise ValueError('Intervention position missing/out of range')
            for p in indices:
                base = updated[0, p, :].clone()
                updated[0, p, :] = edit_tensor(base, spec)
                self.effects.append({'layer': layer_id, 'phase': phase, 'token_index': p, 'operator': spec['operator'], 'strength': spec.get('strength', 0), 'source_position': spec.get('source_position'), 'application_position': spec.get('position', 't_A'), 'vector_representation': spec.get('vector_representation'), 'strength_unit': spec.get('strength_unit'), 'control_matching': spec.get('control_matching'), 'vector_norm': float(np.linalg.norm(spec['vector'])) if 'vector' in spec else None, 'hook_location': self.location, 'object_hash': spec.get('object_hash'), 'delta_norm': float((updated[0, p, :] - base).float().norm().item())})
        if phase == 'prefill' and self.positions:
            self.upstream_capture._make_hook(layer_id)(None, None, updated)
            if updated.is_cuda:
                import torch
                torch.cuda.current_stream(updated.device).synchronize()
            captured = self.upstream_capture._captured.pop(layer_id)
            for name, indices in self.positions.items():
                if indices is None:
                    continue
                if not indices or any((p < 0 or p >= hidden.shape[1] for p in indices)):
                    raise ValueError('Capture position out of range')
                self.captures[layer_id, name] = captured[0, indices, :].float().mean(0).numpy()
        return updated

    def __enter__(self):
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.root_hook, with_kwargs=True))
            for layer_id in self.layer_ids:
                if self.location == 'block_output':

                    def hook(module, inputs, output, li=layer_id):
                        return _replace_hidden(output, self.process(li, _extract_hidden(output)))
                    self.handles.append(self.layers[layer_id].register_forward_hook(hook))
                else:

                    def pre(module, inputs, li=layer_id):
                        return (self.process(li, inputs[0]), *inputs[1:])
                    self.handles.append(self.layers[layer_id].register_forward_pre_hook(pre))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
