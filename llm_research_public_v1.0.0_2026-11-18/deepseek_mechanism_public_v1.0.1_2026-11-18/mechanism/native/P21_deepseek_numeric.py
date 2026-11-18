import hashlib
import inspect
import math
from pathlib import Path
from mechanism.native.A04_runtime import TaskFatal
POLICY_ID = 'deepseek-v2-yarn-mscale-v1'
TRANSFORMERS_VERSION = '4.57.6'
BUILTIN_SOURCE_SHA256 = '3e16232e3c0a0fd0090ddc79d28e9669aa0600258a42712913c008f11927d24d'
ORIGINAL_SOURCE_SHA256 = '7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c'

def scale_values(config):
    if config.model_type != 'deepseek_v2':
        raise TaskFatal('Wrong model type for V2 YaRN correction')
    rope = config.rope_scaling
    if not isinstance(rope, dict) or rope.get('rope_type', rope.get('type')) != 'yarn' or float(rope.get('factor', 0)) != 40.0 or (float(rope.get('mscale_all_dim', 0)) != 0.707) or (float(rope.get('mscale', 0)) != 0.707):
        raise TaskFatal('Pinned DeepSeek-V2-Lite-Chat YaRN configuration changed')
    dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    base = dim ** (-0.5)
    multiplier = (1.0 + 0.1 * float(rope['mscale_all_dim']) * math.log(float(rope['factor']))) ** 2
    return {'qk_head_dim': dim, 'base_scale': base, 'mscale_squared': multiplier, 'corrected_scale': base * multiplier, 'rope_factor': float(rope['factor']), 'mscale_all_dim': float(rope['mscale_all_dim'])}

def apply_yarn_scale(model):
    import transformers
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise TaskFatal('Numeric backport requires Transformers ' + TRANSFORMERS_VERSION)
    source = Path(inspect.getsourcefile(DeepseekV2Attention))
    if hashlib.sha256(source.read_bytes()).hexdigest() != BUILTIN_SOURCE_SHA256:
        raise TaskFatal('Pinned built-in DeepSeek V2 source changed; review before running')
    values = scale_values(model.config)
    layers = model.model.layers
    if len(layers) != model.config.num_hidden_layers:
        raise TaskFatal('DeepSeek layer count mismatch')
    for layer in layers:
        attn = layer.self_attn
        if type(attn) is not DeepseekV2Attention or attn.qk_head_dim != values['qk_head_dim']:
            raise TaskFatal('Unexpected DeepSeek attention class/dimension')
        if not any((math.isclose(attn.scaling, x, rel_tol=1e-14, abs_tol=0) for x in (values['base_scale'], values['corrected_scale']))):
            raise TaskFatal('Unexpected attention scale; refusing an unverified numeric path')
    for layer in layers:
        layer.self_attn.scaling = values['corrected_scale']
    evidence = {'policy_id': POLICY_ID, 'transformers_version': TRANSFORMERS_VERSION, 'builtin_source_sha256': BUILTIN_SOURCE_SHA256, 'official_source_sha256': ORIGINAL_SOURCE_SHA256, 'layer_count': len(layers), 'application': 'instance_scaling_absolute_assignment', 'weights_or_rope_frequencies_changed': False, **values}
    verify_yarn_scale(model, evidence)
    return evidence

def verify_yarn_scale(model, evidence):
    values = scale_values(model.config)
    layers = model.model.layers
    if evidence.get('policy_id') != POLICY_ID or evidence.get('layer_count') != len(layers) or any((evidence.get(k) != v for k, v in values.items())):
        raise TaskFatal('Missing or inconsistent YaRN scale evidence')
    if any((not math.isclose(layer.self_attn.scaling, values['corrected_scale'], rel_tol=1e-14, abs_tol=0) for layer in layers)):
        raise TaskFatal('YaRN attention scale changed after initialization')

def synchronize(model):
    """GPU timing only: synchronize when a CUDA model is available."""
    torch = getattr(model, 'torch', None)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
