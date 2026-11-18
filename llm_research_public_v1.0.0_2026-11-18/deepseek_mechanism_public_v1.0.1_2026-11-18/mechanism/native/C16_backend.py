import copy
from mechanism.native.A04_runtime import TaskFatal, digest
POLICY = {'policy_id': 'llama31-bf16-reduction-off-v1', 'bf16_reduced_precision_reduction': False, 'split_k': 'preserve', 'attention_implementation': 'sdpa', 'attention_backend': 'automatic'}

def enabled(cfg):
    return cfg.get('runtime_backend') is not None

def validate_policy(cfg):
    value = cfg.get('runtime_backend')
    if value != POLICY or type(value.get('bf16_reduced_precision_reduction')) is not bool:
        raise TaskFatal('Approved r8 runtime_backend policy differs')
    if cfg.get('model_id') != 'meta-llama/Llama-3.1-8B-Instruct' or cfg.get('dtype') != 'bfloat16':
        raise TaskFatal('The r8 backend policy requires Llama-3.1-8B BF16')

def state(torch):
    c = torch.backends.cuda
    m = c.matmul
    return {'bf16_reduced_precision_reduction': bool(m.allow_bf16_reduced_precision_reduction), 'bf16_split_k': bool(m.allow_bf16_reduced_precision_reduction_split_k), 'math_reduced_precision_reduction': bool(c.fp16_bf16_reduction_math_sdp_allowed()), 'flash_sdp': bool(c.flash_sdp_enabled()), 'efficient_sdp': bool(c.mem_efficient_sdp_enabled()), 'math_sdp': bool(c.math_sdp_enabled()), 'cudnn_sdp': bool(c.cudnn_sdp_enabled()), 'matmul_fp32_precision': str(m.fp32_precision)}

class BackendGuard:
    """Apply for the model lifetime, verify before each use, restore on close/error."""

    def __init__(self, torch, cfg):
        validate_policy(cfg)
        self.torch = torch
        self.closed = False
        try:
            self.before = state(torch)
        except (AttributeError, TypeError) as exc:
            raise TaskFatal('Required PyTorch BF16/split-K API unavailable; no fallback') from exc
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, self.before['bf16_split_k'])
            self.effective = state(torch)
            expected = {**self.before, 'bf16_reduced_precision_reduction': False}
            if self.effective != expected:
                raise TaskFatal('BF16 policy not applied exactly; split-K/attention must be preserved')
            if not all((self.effective[k] for k in ('flash_sdp', 'efficient_sdp', 'math_sdp', 'cudnn_sdp'))):
                raise TaskFatal('r8 requires automatic SDPA selection; a backend was externally disabled')
        except Exception:
            self.close()
            raise
        self.evidence = {'requested': copy.deepcopy(POLICY), 'before': self.before, 'effective': self.effective, 'torch_version': str(torch.__version__), 'cuda_build': torch.version.cuda, 'scope': 'HFModel lifetime in this Python process', 'whole_model_fp32': False, 'split_k_preserved': True}
        self.evidence['execution_fingerprint'] = digest({k: self.evidence[k] for k in ('requested', 'effective', 'torch_version', 'cuda_build')})

    def check(self):
        if self.closed or state(self.torch) != self.effective:
            raise TaskFatal('Effective runtime backend changed during model lifetime')

    def close(self):
        if self.closed:
            return
        self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (self.before['bf16_reduced_precision_reduction'], self.before['bf16_split_k'])
        self.closed = True
        if state(self.torch) != self.before:
            raise TaskFatal('Backend state restoration failed; stop this process')

def check_identity(identity, cfg, reference=None):
    """Check persisted evidence too: old arrays cannot masquerade as r8 measurements."""
    validate_policy(cfg)
    ev = identity.get('runtime_backend', {})
    effective = ev.get('effective', {})
    if ev.get('requested') != POLICY or effective.get('bf16_reduced_precision_reduction') is not False or ev.get('split_k_preserved') is not True or (effective.get('bf16_split_k') != ev.get('before', {}).get('bf16_split_k')) or (identity.get('attention_implementation') != 'sdpa') or (identity.get('dtype') != 'bfloat16') or (identity.get('model_id') != cfg['model_id']) or (identity.get('resolved_revision') != cfg['revision']):
        raise TaskFatal('Missing/inconsistent r8 model/backend evidence; old results cannot be reused')
    fingerprint = digest({k: ev.get(k) for k in ('requested', 'effective', 'torch_version', 'cuda_build')})
    if ev.get('execution_fingerprint') != fingerprint:
        raise TaskFatal('Runtime backend fingerprint changed')
    if reference is not None:
        check_identity(reference, cfg)
        keys = ('model_id', 'resolved_revision', 'tokenizer_revision', 'template_hash', 'dtype', 'layer_count', 'hook_location', 'attention_implementation')
        if any((identity.get(k) != reference.get(k) for k in keys)) or fingerprint != reference['runtime_backend']['execution_fingerprint']:
            raise TaskFatal('Model/runtime differs from this run runtime-check; start a new run')
    return fingerprint
