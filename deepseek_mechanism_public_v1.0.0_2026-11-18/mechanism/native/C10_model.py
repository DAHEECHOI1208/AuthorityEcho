import gc
import importlib.metadata
import json
import platform
import numpy as np
from mechanism.native.A04_runtime import digest, TaskFatal

class HFModel:

    def __init__(self, cfg):
        from mechanism.native.C16_backend import BackendGuard, enabled
        import torch
        self._backend_guard = None
        self.torch = torch
        try:
            if enabled(cfg):
                self._backend_guard = BackendGuard(torch, cfg)
            self._initialize(cfg)
        except Exception:
            self.close()
            raise

    def _initialize(self, cfg):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        for key in ('model_id', 'revision', 'layers_path', 'max_input_tokens'):
            if not cfg.get(key):
                raise ValueError('Missing model.' + key)
        self.cfg = cfg
        validate_extraction_mode(cfg)
        self.torch = torch
        dtype = getattr(torch, cfg.get('dtype', 'float32'))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg.get('tokenizer_revision') or cfg['revision'], use_fast=True, trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise ValueError('Fast tokenizer offset mapping required; no guessed positions')
        from mechanism.native.P19_deepseek import enabled as is_deepseek, validate as validate_deepseek
        loaded = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=cfg['revision'], torch_dtype=dtype, device_map=cfg.get('device_map', 'auto'), trust_remote_code=False, **{'output_loading_info': True} if is_deepseek(cfg) else {}, **{'attn_implementation': cfg.get('attention_implementation', 'sdpa')} if self._backend_guard or cfg.get('attention_implementation') else {})
        if is_deepseek(cfg):
            self.model, loading_info = loaded
            if any((loading_info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))):
                raise TaskFatal('Checkpoint loading mismatch: ' + str(loading_info))
        else:
            self.model = loaded
        self.model.eval()
        if is_deepseek(cfg):
            validate_deepseek(cfg)
            cls = type(self.model)
            if cls.__name__ != 'DeepseekV2ForCausalLM' or not cls.__module__.startswith('transformers.models.deepseek_v2.'):
                raise TaskFatal('Expected built-in Transformers DeepseekV2ForCausalLM')
            if self.model.config._attn_implementation != 'sdpa':
                raise TaskFatal('DeepSeek SDPA not applied')
            from mechanism.native.P21_deepseek_numeric import apply_yarn_scale, POLICY_ID
            if cfg.get('attention_scale_policy') != POLICY_ID:
                raise TaskFatal('Explicit corrected attention-scale policy required')
            self._deepseek_numeric = apply_yarn_scale(self.model)
        self.layers = self.model
        for part in cfg['layers_path'].split('.'):
            self.layers = getattr(self.layers, part)
        if not self.tokenizer.chat_template:
            raise ValueError('Native chat template missing')
        self.identity = {'backend': 'huggingface', 'model_id': cfg['model_id'], 'revision': cfg['revision'], 'resolved_revision': getattr(self.model.config, '_commit_hash', None), 'tokenizer_revision': cfg.get('tokenizer_revision') or cfg['revision'], 'template_hash': digest(self.tokenizer.chat_template), 'dtype': cfg.get('dtype', 'float32'), 'layers_path': cfg['layers_path'], 'layer_count': len(self.layers), 'hook_location': cfg.get('hook_location', 'block_output'), 'quantization': 'none', 'generation': cfg.get('generation', {}), 'model_generation_defaults': self.model.generation_config.to_dict()}
        if is_deepseek(cfg):
            from mechanism.native.P19_deepseek import identity as deepseek_identity
            self.identity.update(deepseek_identity(self.tokenizer))
            self.identity['attention_implementation'] = self.model.config._attn_implementation
            self.identity['checkpoint_loading_info'] = loading_info
            self.identity['attention_scale_policy'] = cfg['attention_scale_policy']
            self.identity['numeric_backport'] = self._deepseek_numeric
        if cfg.get('final_answer_policy') is not None:
            self.identity['final_answer_policy'] = cfg['final_answer_policy']
        if self._backend_guard:
            self._backend_guard.check()
            self.identity['runtime_backend'] = self._backend_guard.evidence
            self.identity['attention_implementation'] = getattr(self.model.config, '_attn_implementation', None)
            from mechanism.native.C16_backend import check_identity
            check_identity(self.identity, cfg)
        if cfg.get('quantization') not in (None, 'none'):
            raise ValueError('This release supports unquantized HF inference only')

    def serialize(self, row):
        from mechanism.native.P19_deepseek import enabled, serialize
        if enabled(self.cfg):
            return serialize(row, self.cfg, self.tokenizer)
        messages = json.loads(json.dumps(row['messages']))
        if self.cfg.get('tool_arguments_format', 'object') == 'object':
            for m in messages:
                for c in m.get('tool_calls', []):
                    v = c['function']['arguments']
                    if isinstance(v, str):
                        c['function']['arguments'] = json.loads(v)
        from mechanism.native.P17_final_answer import template_kwargs
        kwargs = template_kwargs(row, self.cfg)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        native = self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        if ids != native:
            raise ValueError('Native template IDs differ from offset encoding')
        if len(ids) > int(self.cfg['max_input_tokens']):
            raise TaskFatal('Input exceeds max_input_tokens; stop and inspect template/input; truncation is forbidden')
        from mechanism.native.F10_activations import resolve_positions
        fragments = None
        if self.cfg.get('model_id') == 'meta-llama/Llama-3.1-8B-Instruct':
            user = [m['content'] for m in row['messages'] if m['role'] == 'user'][-1]
            payload = row.get('tool_payload')
            document = row.get('document', '')
            once = json.dumps(document, ensure_ascii=False)[1:-1] if document else None
            fragments = {'user': user.strip(), 'tool_payload': json.dumps(payload, ensure_ascii=False) if payload is not None else None, 'document': json.dumps(once, ensure_ascii=False)[1:-1] if once is not None else None}
        positions = resolve_positions(text, encoded['offset_mapping'], row, fragments)
        return {'text': text, 'input_ids': ids, 'attention_mask': [1] * len(ids), 'positions': positions, 'offset_mapping': encoded['offset_mapping'], 'template_hash': self.identity['template_hash']}

    def tensors(self, serialized):
        if hasattr(self, '_deepseek_numeric'):
            from mechanism.native.P21_deepseek_numeric import verify_yarn_scale
            verify_yarn_scale(self.model, self._deepseek_numeric)
        if getattr(self, '_backend_guard', None):
            self._backend_guard.check()
        device = self.model.get_input_embeddings().weight.device
        ids = self.torch.tensor([serialized['input_ids']], device=device)
        return {'input_ids': ids, 'attention_mask': self.torch.ones_like(ids)}

    def session(self, serialized, layers, positions, interventions=None):
        if getattr(self, '_backend_guard', None):
            self._backend_guard.check()
        from mechanism.native.C11_hooks import HookSession
        specs = []
        for s in interventions or []:
            s = dict(s)
            if 'token_indices' not in s:
                s['token_indices'] = serialized['positions'].get(s.get('position', 't_A'))
            if s['token_indices'] is None:
                raise ValueError('No intervention position')
            specs.append(s)
        all_layers = sorted(set(layers) | {int(s['layer']) for s in specs})
        return HookSession(self.model, self.layers, all_layers, positions, self.identity['hook_location'], specs, len(serialized['input_ids']))

    def generate(self, row, interventions=None):
        serialized = self.serialize(row)
        inputs = self.tensors(serialized)
        gen = dict(self.cfg.get('generation', {}))
        gen.setdefault('max_new_tokens', 512)
        gen.setdefault('do_sample', False)
        if gen.get('num_beams', 1) != 1 or gen.get('num_return_sequences', 1) != 1:
            raise ValueError('Reference runtime requires one beam and one return sequence')
        gen['use_cache'] = True
        from transformers import GenerationConfig
        defaults = self.model.generation_config.to_dict()
        effective = GenerationConfig.from_dict(defaults)
        model_kwargs = effective.update(**gen)
        generation_settings = {'model_defaults': defaults, 'overrides': dict(gen), 'effective': effective.to_dict(), 'non_config_kwarg_names': sorted(model_kwargs), 'recording_stage': 'resolved_generation_config_passed_to_generate'}
        if 'seed' in self.cfg:
            self.torch.manual_seed(int(self.cfg['seed']) + int(row.get('repeat_id', 0)))
        with self.session(serialized, [], {}, interventions) as session:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, generation_config=effective, **model_kwargs)
        tokens = output[0, len(serialized['input_ids']):].tolist()
        eos = gen.get('eos_token_id', self.model.generation_config.eos_token_id)
        eos = [eos] if isinstance(eos, int) else eos or []
        stopped = bool(tokens and tokens[-1] in eos)
        defaults = self.model.generation_config.to_dict()
        other_stops = any((gen.get(k, defaults.get(k)) for k in ('stop_strings', 'max_time', 'stopping_criteria')))
        reason, truncated = classify_termination(tokens, eos, gen['max_new_tokens'], other_stops)
        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        from mechanism.native.P17_final_answer import enabled, output_form
        form = output_form(response, tokens, self.cfg['model_id']) if enabled(self.cfg) else {}
        return {**row, **form, 'response_text': response, 'response_token_ids': tokens, 'response_token_count': len(tokens), 'generation_settings': generation_settings, 'technical_status': 'ok' if response.strip() else 'empty_response', 'stop_reason': reason, 'truncated': truncated, 'ended_with_eos': stopped, 'termination_settings': {'eos_token_id': eos, 'max_new_tokens': gen['max_new_tokens'], 'max_time': gen.get('max_time', defaults.get('max_time')), 'stop_strings': gen.get('stop_strings', defaults.get('stop_strings')), 'other_stopping_criteria': bool(other_stops)}, 'serialized': serialized, 'model': self.identity, 'synthetic': False, 'interventions': interventions or [], 'effects': session.effects}

    def extract(self, row, layers, positions, interventions=None):
        serialized = self.serialize(row)
        span = {p: serialized['positions'].get(p) for p in positions}
        with self.session(serialized, layers, span, interventions) as session:
            with self.torch.inference_mode():
                self.model(**self.tensors(serialized), use_cache=False)
        hidden = int(self.model.config.hidden_size)
        arr = np.full((len(layers), len(positions), hidden), np.nan, dtype=np.float32)
        for li, l in enumerate(layers):
            for pi, p in enumerate(positions):
                if (l, p) in session.captures:
                    arr[li, pi] = session.captures[l, p]
        return (arr, {'model': self.identity, 'synthetic': False, 'layers': layers, 'positions': positions, 'serialized': serialized, 'effects': session.effects, 'aggregation': 'within_input_token_mean', 'capture_phase': 'prefill', 'extraction_mode': 'native', 'layer_indexing': 'zero_based_decoder_block', 'capture_implementation': 'AB SelectiveLayerCapture._make_hook + NEW position adapter', 'feature_definitions': {p: {'token_indices': span[p], 'aggregation': 'token' if span[p] and len(span[p]) == 1 else 'within_input_token_mean', 'module_paths': [self.cfg['layers_path'] + '.' + str(l) for l in layers], 'hook_location': self.identity['hook_location']} for p in positions}})

    def close(self):
        try:
            if hasattr(self, 'model'):
                del self.model
            gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        finally:
            if getattr(self, '_backend_guard', None):
                self._backend_guard.close()

def validate_extraction_mode(cfg):
    if cfg.get('extraction_mode', 'native') != 'native':
        raise TaskFatal('LE L+1 original compatibility is not implemented; only native extraction is supported')
    if cfg.get('le_original_compatibility'):
        raise TaskFatal('LE original compatibility is not supported in v1.2')

def classify_termination(tokens, eos, max_new_tokens, other_stops=False):
    if tokens and tokens[-1] in eos:
        return ('eos', False)
    if not other_stops and len(tokens) == max_new_tokens:
        return ('max_new_tokens', True)
    return ('unknown_stopping_criteria', None)
