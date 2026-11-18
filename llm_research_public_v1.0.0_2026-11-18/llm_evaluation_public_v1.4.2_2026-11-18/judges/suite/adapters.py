"""Connect the packaged judge builders and response parsers."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math

from .common import ROOT, read_json, digest, verify_sources
from .validation import CONFLICT, strongreject_raw_numbers
from .diagnostics import strongreject_diagnostic

FILES = {'harmbench': 'judge_harmbench_hf.py', 'harmbench_full': 'judge_harmbench_full.py',
         'strongreject': 'judge_llama3_70b_strongreject.py', 'together': 'judge_together.py'}


@lru_cache(None)
def legacy(name):
    path = ROOT / 'legacy' / name / FILES[name]
    spec = importlib.util.spec_from_file_location('packaged_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions():
    """Luna first, then the three remaining judge profiles.

    Luna and DeepSeek share the original research rubric, builder and parser.
    component identifies reusable source code; provider controls API routing.
    """
    oc = read_json(ROOT / 'configs/gpt56_luna.json')
    model = oc['models'][0]
    defs = {'gpt56_luna_research': {
        'id': 'gpt56_luna_research', 'component': 'together', 'config': oc,
        'model_spec': model, 'model': model['model_id'], 'revision': None,
        'provider': 'openai'}}
    for judge_id, component in [('harmbench_full', 'harmbench'),
                                ('strongreject', 'strongreject')]:
        cfg = read_json(ROOT / 'legacy' / component / ('config_full.json' if component == 'harmbench' else 'config.json'))
        model = cfg.get('model_id') or cfg['models'][0]['model_id']
        revision = cfg.get('model_revision') or cfg['models'][0]['model_revision']
        defs[judge_id] = {'id': judge_id, 'component': component, 'config': cfg,
                         'model': model, 'revision': revision, 'provider': 'hf'}
    tc = read_json(ROOT / 'legacy/together/config.json')
    for alias, judge_id in [('deepseek', 'deepseek_research')]:
        model = next(m for m in tc['models'] if m['alias'] == alias)
        defs[judge_id] = {'id': judge_id, 'component': 'together', 'config': tc, 'model_spec': model,
                         'model': model['model_id'], 'revision': None, 'provider': 'together'}
    return defs


class Adapter:
    def __init__(self, judge_id, tokenizer=None):
        self.definition = definitions()[judge_id]
        self.id = judge_id
        self.component = self.definition['component']
        self.cfg = self.definition['config']
        self.module = legacy(self.component)
        self.tokenizer = tokenizer
        self.model = self.definition['model']
        self.kind = 'research' if 'research' in judge_id else ('harmbench' if judge_id == 'harmbench_full' else judge_id)
        folder = ROOT / 'legacy' / self.component
        if self.kind == 'harmbench':
            # The full add-on's original hash guard must still pass.
            addon = legacy('harmbench_full')
            self.module = addon.load_runner(folder / FILES['harmbench'])
            self.cfg = addon.check_full_config(self.module, folder / 'config_full.json')
            self.prompt = read_json(folder / 'assets/llama2_cls_prompt.json')['prompt']
        elif self.kind == 'research':
            self.prompt = (folder / 'judge_prompt.txt').read_text(encoding='utf-8').strip()
        else:
            self.prompt = (folder / self.cfg['rubric_file']).read_bytes().decode('utf-8')
            if digest(self.prompt) != self.cfg['rubric_sha256']:
                raise ValueError('Original rubric hash mismatch')
        self.system = (folder / 'sr_judge_system.txt').read_text(encoding='utf-8') if self.kind == 'strongreject' else None

    def build(self, question, answer, served_model=None):
        served_model = served_model or self.model
        info = {'response_mode': 'full', 'response_truncated': False,
                'original_response_sha256': digest(answer), 'evaluated_response_sha256': digest(answer),
                'prompt_sha256': digest(self.prompt), 'token_check': 'server_only'}
        if self.kind == 'harmbench':
            if self.tokenizer is None:
                raise ValueError('HarmBench tokenizer required')
            task = self.module.make_task({'question_id': '_', 'source_row': 0, 'question': question,
                                        'responses': {'C0': answer}}, 'C0', self.cfg, self.tokenizer, self.prompt)
            payload = self.module.payload_for(task, served_model)
            info.update(prompt_tokens=task['prompt_tokens'], output_reserved=1, context_limit=2048,
                        token_check='pinned_harmbench', preflight_status=task['status'])
            route = '/completions'
        elif self.component == 'strongreject':
            spec = {'config': self.cfg, 'prompt': self.prompt, 'system_prompt': self.system}
            payload = self.module.make_payload(spec, self.cfg['models'][0], question, answer, served_model)
            route = '/chat/completions'
        else:
            spec = {'config': self.cfg, 'prompt': self.prompt}
            payload = self.module.make_payload(spec, self.definition['model_spec'], question, answer)
            route = '/chat/completions'
        info['request_input_sha256'] = digest(payload.get('messages', payload.get('prompt')))
        return route, payload, info

    def interpret(self, response, phase='judge'):
        """Preserve original judgement, add independent analysis eligibility."""
        meta = {'phase': phase, 'judge_alias': self.id}
        if self.component in ('strongreject', 'together'):
            result = self.module.interpret(meta, response)
        else:
            result = self._interpret_hf(response)
        result = dict(result)
        if phase == 'smoke' and self.component == 'harmbench' and result.get('status') == 'OK':
            if result.get('label') != 'No':
                result.update(status='SMOKE_UNEXPECTED_LABEL', error='Unexpected label for the smoke refusal example')
        result['legacy_status'] = result.get('status')
        result['analysis_valid'] = result.get('status') == 'OK'
        result['analysis_exclusion'] = None if result['analysis_valid'] else result.get('status')
        if result.get('status') != 'OK':
            result['legacy_label'] = result.get('label')
            result['label'] = None
        if self.kind == 'strongreject' and result.get('status') == 'OK':
            score = result['score']
            if not result.get('in_range') or not math.isfinite(score) or not 0 <= score <= 1:
                result.update(analysis_valid=False, analysis_exclusion='STRONGREJECT_OUT_OF_RANGE',
                              quality_warning='Original score/items retained; excluded from every analysis denominator.')
        # The legacy summary fields had 1,000/4,000-character limits. Raw logs and this field are full.
        try:
            data = json.loads(response.get('raw_body_text', ''))
            choice = data['choices'][0]
            result['raw_judge_output'] = choice.get('text') if self.kind == 'harmbench' else choice.get('message', {}).get('content')
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            result['raw_judge_output'] = None
        if self.kind == 'strongreject':
            result.update(strongreject_raw_numbers(result['raw_judge_output'], result))
            if result.get('status') == 'OK' and result['raw_numeric_valid'] is False:
                result['analysis_valid'] = False
                conflict = any(issue['reason'] == CONFLICT for issue in result['raw_numeric_issues'])
                # Conflict takes display priority; all numeric reasons and the
                # original parsed items/score/status remain available.
                result['analysis_exclusion'] = (CONFLICT if conflict else
                                               result['analysis_exclusion'] or 'STRONGREJECT_RAW_NUMERIC_INVALID')
                warning = ('Conflicting rubric values' if conflict else 'Raw rubric numeric value invalid')
                warning += '; original score/items retained and excluded from analysis.'
                result['quality_warning'] = (result.get('quality_warning', '') + ' ' + warning).strip()
            result.update(strongreject_diagnostic(result))
        return result

    def _interpret_hf(self, response):
        if response.get('network_error'):
            return {'status': response.get('error_status', 'NETWORK_ERROR'), 'error': response['network_error']}
        http = response.get('http_status')
        if http != 200:
            return {'status': 'HTTP_ERROR', 'http_status': http, 'error': response.get('raw_body_text', '')}
        try:
            data = json.loads(response['raw_body_text'])
            if not isinstance(data.get('choices'), list) or len(data['choices']) != 1:
                return {'status': 'PROTOCOL_ERROR'}
            choice = data['choices'][0]
            finish = choice.get('finish_reason')
            extra = {'finish_reason': finish, 'returned_model': data.get('model'), 'usage': data.get('usage'),
                     'response_id': data.get('id')}
            if self.kind == 'harmbench':
                # Execute the packaged judge_one's Yes/No normalization unchanged.
                class ReceivedClient:
                    def request(inner, *args, **kwargs):
                        return data, {'http_status': http}
                task = {key: None for key in ('question_id', 'condition', 'question_sha256', 'response_sha256', 'prompt_sha256')}
                task.update(prompt='', response_truncated=False)
                return self.module.judge_one(ReceivedClient(), task, self.model, 'offline-parser')
            message = choice.get('message') or {}
            if finish in ('length', 'max_tokens'):
                return {**extra, 'status': 'JUDGE_TRUNCATED'}
            if finish not in ('stop', 'eos', 'eos_token'):
                return {**extra, 'status': 'JUDGE_INCOMPLETE'}
            if message.get('refusal'):
                return {**extra, 'status': 'JUDGE_REFUSAL'}
            raw = message.get('content')
            if not isinstance(raw, str) or not raw.strip():
                return {**extra, 'status': 'JUDGE_EMPTY_OUTPUT'}
            return {**extra, 'status': 'OK', 'label': self.module.parse_label(raw)}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            return {'status': 'INVALID_OUTPUT', 'error': str(exc)}


def adapters_for(ids, run_dir, tokenizers=None):
    verify_sources()
    tokenizers = tokenizers or {}
    result = {}
    for judge_id in ids:
        tok = tokenizers.get(judge_id)
        if tok is None and judge_id == 'harmbench_full':
            tok = legacy('harmbench').tokenizer()
        result[judge_id] = Adapter(judge_id, tok)
    return result
