"""Save the exact benchmark prompt and public protocol without credentials."""
import hashlib
import json
from pathlib import Path
import platform
from run_human_llm import PROMPT

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/human_llm'
    (out/'prompt.txt').write_text(PROMPT)
    protocol={
        'tasks':['pressure_detection','intensity_prediction','strategy_recognition','dimension_prediction'],
        'test_n':634,'train_n':3088,'validation_n':344,'seed':2026,
        'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'annotation_sha256':hashlib.sha256((ROOT/'data/human/annotations.jsonl').read_bytes()).hexdigest(),
        'split_sha256':hashlib.sha256((ROOT/'data/human/splits.json').read_bytes()).hexdigest(),
        'models':{
            'gpt':{'requested_model':'gpt-6-astra','api':'responses','reasoning_effort':'low','temperature':'provider_default','max_output_tokens':1600,
                   'instructions':'Follow the supplied annotation task. Return only the requested JSON object.'},
            'qwen':{'requested_model':'qwen3.8-max','api':'chat_completions','enable_thinking':False,'temperature':0,'max_tokens':1600},
            'deepseek':{'requested_model':'deepseek-flash','api':'chat_completions','thinking':{'type':'disabled'},'temperature':0,'max_tokens':1600}},
        'invalid_outputs':'Preserve raw values; evaluate tasks independently and report coverage. No generated-answer retries.',
        'transport_retries':'GPT resumed run: up to 3 retries after initial request, with 5/10/15 second delays.',
        'field_aliases':{'D3_Logical_False_Dilemma':'D3_Logical_Trap_False_Dilemma'},
        'json_normalization':'Close exactly one missing outer brace when the final field is complete and the decoder error is at end-of-input; never supply or change label values.',
        'excluded_runs':'Development smoke tests and an aborted GPT default-reasoning run; not included in this supplement.',
        'dependencies':{'python':platform.python_version()}}
    for module in ['numpy','scipy','sklearn']:
        protocol['dependencies'][module]=__import__(module).__version__
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(out/'protocol.json')


if __name__=='__main__':main()
