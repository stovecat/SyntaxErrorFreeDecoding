import torch
import pickle
from transformers import (AutoTokenizer, AutoModelWithLMHead, T5ForConditionalGeneration, 
                            XGLMForCausalLM, AutoModelForCausalLM, GPTNeoForCausalLM, 
                            GPTJForCausalLM, GPT2Tokenizer, GPTNeoForCausalLM)

def check_syntax_error(_code, has_eos=False):
    code = _code
    if len(code.split("\n")) > 1:
        code = "\n".join([l for l in code.split("\n") if l != ""])
    if code != "" and code[-1] == "\n":
        code = code[:-1]
    try:
        compile(code, 'test.py', mode='exec')
    except Exception as e:
        if has_eos and "TabError" == e.__class__.__name__:
            return None
        elif not has_eos and ("EOF" in str(e) or \
                               "EOL" in str(e) or \
                               ("expected an indented block" in str(e) and e.lineno == len(code.split('\n'))) or \
                               "TabError" == e.__class__.__name__ or \
                                e.lineno == len(code.split('\n')) and e.__class__.__name__ == "SyntaxError" and \
                                e.text is not None and len(e.text) > 0 and e.text[-1]=='\n'):
            return None
        return e
    return None

def get_index_by_value(t, v):
    return (t == v).nonzero(as_tuple=True)[0]


# +
def load_pkl(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def dump_pkl(path, data):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


# +
import random
import numpy as np
import os

def set_seed(seed, n_gpu, FULL_REPRODUCIBILITY=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if FULL_REPRODUCIBILITY:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


# -

def get_model(name):
    '''
    Available pre-trained model list:
        [~ 1B]
        ( 60M) Salesforce/codet5-small
        (110M) https://github.com/microsoft/PyCodeGPT
        (125M) EleutherAI/gpt-neo-125M
        (160M) polycoder-160M https://zenodo.org/record/6363556/files/160M-150K.tar
        (220M) Salesforce/codet5-base
        (350M) Salesforce/codegen-350M-multi
        (350M) Salesforce/codegen-350M-mono
        (400M) polycoder-400M https://zenodo.org/record/6363556/files/0-4B-150K.tar
        (770M) Salesforce/codet5-large
        (770M) Salesforce/codet5-large-ntp-py
        
        [< 3B]
        (1B)   facebook/incoder-1B
        (1.3B) EleutherAI/gpt-neo-1.3B
        (1.3B) #GPT3_XL https://mystic.the-eye.eu/public/AI/gptneo-release/GPT3_XL/
        (1.3B) gpt-neo-1.3B-APPS https://drive.google.com/file/d/1XW1Od9L-5l9zXl1HUCyER5pS9zQTbIvU/view
        (1.5B) codeparrot/codeparrot
        (2B)   Salesforce/codegen-2B-multi
        (2B)   Salesforce/codegen-2B-mono
        (2.7B) EleutherAI/gpt-neo-2.7B
        (2.7B) polycoder-2.7B https://zenodo.org/record/6363556#.YuDOUOxByJE
        (2.7B) gpt-neo-2.7B-APPS https://drive.google.com/file/d/1XW1Od9L-5l9zXl1HUCyER5pS9zQTbIvU/view
        (2.7B) gpt3-2.7B https://mystic.the-eye.eu/public/AI/gptneo-release/GPT3_2-7B/
        
        [< 10B]
        (6B)   EleutherAI/gpt-j-6B
        (6.1B) Salesforce/codegen-6B-multi
        (6.1B) Salesforce/codegen-6B-mono
        (6.7B) facebook/incoder-6B
        
        [> 10B]
        (16.1B) Salesforce/codegen-16B-mono
        (16.1B) Salesforce/codegen-16B-multi
        (20B)   EleutherAI/gpt-neox-20b
        
    '''
    local_ckpt_list = ['gpt3-xl-1.3B', 'polycoder-160M', 'polycoder-400M', 'polycoder-2.7B', 
                       'gpt-neo-1.3B-APPS', 'gpt-neo-2.7B-APPS', 'gpt3-2.7B']
    
    if "PUBLIC" in name or "the-Pile" in name:
        from mutransformers.models.gptj.modeling_gptj import GPTJForCausalLM
        from mup import set_base_shapes
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        ckpt = torch.load(name)
        ckpt['model_config'].exponential_decay_length_penalty = None
#         model = GPTJForCausalLM.from_pretrained(ckpt['model_config'])#, device_map="auto", torch_dtype=torch.half)
#         model = GPTJForCausalLM.from_pretrained(ckpt['model_config'])
        model = GPTJForCausalLM(ckpt['model_config'])
        #model.parallelize()
        
        if "./pretrained/gptj-350M" in name:
            set_base_shapes(model, "./pretrained/gptj-350M-PUBLIC/gptj_embed256_layer20.bsh")
        elif "./pretrained/gptj-770M" in name:
            set_base_shapes(model, "./pretrained/gptj-770M-p-PUBLIC/gptj_embed256_layer36.bsh")
            language_list=['JavaScript', 'GO', 'PHP', 'C', 'CSS', 'Markdown', 'C++', 'Batchfile', 'Java', 'SQL', 'Python', 'HTML', 'Shell', 'C#', 'Ruby', 'TypeScript', 'Rust', 'Scala', 'CMake', 'TeX', 'Lua', 'Julia', 'Makefile', 'FORTRAN', 'PowerShell', 'Assembly', 'Perl', 'Dockerfile', 'Visual Basic', 'Haskell', 'go', 'java', 'javascript', 'php', 'python', 'ruby', '']
            language_list = set([x.lower() for x in language_list])
            spc_token_list = ["<|{}|>".format(x) if x != "" else "<|unknown_language|>" for x in language_list]
            tokenizer.add_tokens(spc_token_list, special_tokens=True)
        model.load_state_dict(ckpt['model'])
        assert model is not None
        model.half()
        return model, tokenizer
    
    if "./apps/train/finetuned/EleutherAI_gpt-neo-2.7B/" in name or name in ['./pretrained/gpt-neo-2.7B-APPS', './pretrained/gpt-neo-1.3B-APPS']:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPTNeoForCausalLM.from_pretrained(name)
        model.half()
        return model, tokenizer
    elif "./apps/train/finetuned/Salesforce_codegen-350M" in name:
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model = AutoModelForCausalLM.from_pretrained(name)
        #from apps.train.model import Model
        #model = Model.load_state_dict(torch.load(name))
        model.half()
        return model, tokenizer
    elif "./apps/train/finetuned/Salesforce_codegen-2B" in name:
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
        model = AutoModelForCausalLM.from_pretrained(name)
        #from apps.train.model import Model
        #model = Model.load_state_dict(torch.load(name))
        model.half()
        return model, tokenizer
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    model = None
    # InCoder uses XGLMForCausalLM
    if 'facebook/incoder' in name:
        model = XGLMForCausalLM.from_pretrained(name)
    # CodeGen uses AutoModelForCausalLM
    elif 'Salesforce/codegen' in name:
        model = AutoModelForCausalLM.from_pretrained(name)#, device_map="auto", torch_dtype=torch.half)#, torch_dtype=torch.half)        
    elif 'EleutherAI/gpt-neo' in name:
        model = GPTNeoForCausalLM.from_pretrained(name)
    elif 'EleutherAI/gpt-j' in name:
        from transformers import GPTJForCausalLM
        model = GPTJForCausalLM.from_pretrained(name)
    # CodeT5 uses T5ForConditionalGeneration
    elif 'Salesforce/codet5' in name:
        model = T5ForConditionalGeneration.from_pretrained(name)
    elif name in local_ckpt_list:
        model = AutoModelWithLMHead.from_pretrained('./pretrained/'+name)
    else:
        model = AutoModelWithLMHead.from_pretrained(name)
    
    assert model is not None
    model.half()
    return model, tokenizer


import os
import numpy as np
from evaluate import load
from datasets import load_dataset
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def _evaluate(candidates, test_cases, k=[1, 5, 10]):
    '''
    candidates = [["def add(a,b): return a*b", "def add(a, b): return a+b"]]
    test_cases = ["assert add(2,3)==5"]
    '''
    code_eval_metric = load("code_eval")
    pass_at_k, results = code_eval_metric.compute(references=test_cases, predictions=candidates, k=k)
    return pass_at_k, results  


def get_test_cases(human_eval):
    test_cases_list = []
    for test in human_eval['test']:
        test_cases_list.append(test['test']+f"\ncheck({test['entry_point']})")
    return test_cases_list


def evaluate_human_eval(human_eval, cand_list, test_cases_list, k_list=[1, 5, 10], estimate=True):
    _, results = _evaluate(cand_list, test_cases_list)
    # following (Chen et al. 2021)
    pass_at_k_chen = {k: 0 for k in k_list} 
    # following (Kulal et al. 2019)
    pass_at_k_kulal = {k: 0 for k in k_list}
    for i in range(len(results)):
        r = [r[1]['passed'] for r in results[i]]
        for k in k_list:
            pass_at_k_chen[k] += 1. if True in r else 0.
            pass_at_k_kulal[k] += 1. if True in r[:k] else 0.
    for k in k_list:
        pass_at_k_chen[k] /= len(results)
        pass_at_k_kulal[k] /= len(results)
        #if estimate: 
        print(f'pass@{k} (chen): {round(pass_at_k_chen[k]*100, 2)}%')
        #else:
        print(f'pass@{k} (kulal): {round(pass_at_k_kulal[k]*100, 2)}%')
    
    return pass_at_k_chen, pass_at_k_kulal


def truncate_before_pattern(string, patterns):
    for p in patterns:
        string = string.split(p)[0]
    return string


def get_inference_input(d, lang='python'):
    comment_start, comment_end = None, None
    if lang == 'python':
        comment_start = "'''\n"
        comment_end = "'''\n"
    elif lang == 'cpp':
        comment_start = '/*\n'
        comment_end = '*/\n'
    else:
        raise NotImplementedError
        
    input = f"{comment_start}RATING: {d['cf_rating']}\n"
    input += f"TAGS: {', '.join(d['cf_tags'])}\n"
    input += f"LANGUAGE IS {lang}\n"
    input += f"CORRECT SOLUTION\n"
    input += f"{d['description']}\n{comment_end}"
    return input


def get_inference_input(d, lang='python', metadata=True, description_option='single'):
    input = ""
    if description_option == 'single':
        if lang == 'python':
            comment = "# "
        elif lang in ['cpp', 'c']:
            comment = '// '
        else:
            raise NotImplementedError
        if metadata:
            input += f"{comment}RATING: {d['cf_rating']}\n"
            input += f"{comment}TAGS: {', '.join(d['cf_tags'])}\n"
            input += f"{comment}LANGUAGE IS {lang}\n"
            input += f"{comment}CORRECT SOLUTION\n"
        for line in d['description'].split("\n"):
            input += f"{comment}{line}"
        input +="\n"
    elif description_option == 'multi':    
        comment_start, comment_end = None, None
        if lang == 'python':
            comment_start = "'''\n"
            comment_end = "'''\n"
        elif lang in ['cpp', 'c']:
            comment_start = '/*\n'
            comment_end = '*/\n'
        else:
            raise NotImplementedError
        if metadata:    
            input += f"{comment_start}RATING: {d['cf_rating']}\n"
            input += f"TAGS: {', '.join(d['cf_tags'])}\n"
            input += f"LANGUAGE IS {lang}\n"
            input += f"CORRECT SOLUTION\n"
        input += f"{d['description']}\n{comment_end}"
    elif description_option == 'none':
        if metadata:
            input += f"RATING: {d['cf_rating']}\n"
            input += f"TAGS: {', '.join(d['cf_tags'])}\n"
            input += f"LANGUAGE IS {lang}\n"
            input += f"CORRECT SOLUTION\n"
        input += f"{d['description']}\n"
    return input


# +
import random

def get_1_shot_input(train, d_eval, tokenizer, max_input_length=1024, lang='python'):
    desciption = get_inference_input(d_eval, lang)
    max_1_shot_length = max(0, max_input_length-len(tokenizer.tokenize(desciption)))
    if max_1_shot_length == 0:
        return tokenizer.decode(tokenizer.encode(desciption)[:max_input_length])
    
    input = None
    n_iter = 0
    max_recursion = 1000
    while input is None:
        if n_iter > max_recursion: # 
            input = ''
            break
        n_iter += 1
        d_sample = random.choice(train)
        py_sol_indices = [i for i, v in enumerate(d_sample['solutions']['language']) if v == 3]
        if len(py_sol_indices) == 0:
            continue
        input = get_inference_input(d_sample, lang)
        sample_ans = d_sample['solutions']['solution'][random.choice(py_sol_indices)]
        input += f"{sample_ans}\n"
        if len(tokenizer.tokenize(input)) > max_1_shot_length:
            input = None
            continue
    input += desciption
    
    assert len(tokenizer.tokenize(input)) <= max_input_length

    return input


# +
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm

def get_tests(problem, example_tests=0):
    tests = {}
    for key in ['input', 'output']:
        tests[key+'s'] = problem['public_tests'][key]
        if example_tests == 0:
            tests[key+'s'].extend(problem['private_tests'][key])
            tests[key+'s'].extend(problem['generated_tests'][key])
    return tests

# def evaluate_code_contests(code_contests, candidates_list, k_list=[1]):
#     passed = {}
#     for i in tqdm(range(len(code_contests))):
#         if i not in py_sols:
#             continue
#         d = code_contests[i]
#         tests = get_tests(d)
#         passed_list = [None] * len(tests['inputs'])
        
#         if len(candidates_list[i]) != 1 or k_list != [1]:
#             raise NotImplementedError
            
#         for n, sol in candidates_list[i]:
#             code_fn = f"../code_contests/preds/{i}/sol_{n}.py"
#             os.makedirs("/".join(code_fn.split("/")[:-1]), exist_ok=True)
#             with open(code_fn, 'w', encoding='utf-8') as fp:
#                 fp.write(sol)
#             for j in range(len(tests['inputs'])):
#                 input_fn = f"../code_contests/inputs/{i}/input_{j}.txt"
#                 output_fn = f"../code_contests/outputs/{i}/output_{j}.txt"
#                 os.makedirs("/".join(input_fn.split("/")[:-1]), exist_ok=True)
#                 os.makedirs("/".join(output_fn.split("/")[:-1]), exist_ok=True)
#                 with open(input_fn, 'w', encoding='utf-8') as fp:
#                     fp.write(tests['inputs'][j])
#                 os.system(f"python {code_fn} < {input_fn} > {output_fn}")
#                 with open(output_fn, 'r', encoding='utf-8') as fp:
#                     stdout_data = fp.read()
#                 is_passed = stdout_data.lstrip().rstrip() == tests['outputs'][j].lstrip().rstrip()
#                 passed_list[j] = (is_passed, tests['inputs'][j], tests['outputs'][j], stdout_data)
#                 if not is_passed:
#                     print(f'Solution failed in {i}th problem')
#                     break
#             passed[i] = passed_list
    
#     # Count the number of all-passed predictions
#     cnt = 0
#     for item in passed.values():
#         flag = True
#         for v in item:
#             if not v[0]:
#                 flag = False
#                 break
#         if flag:
#             cnt += 1
            
#     pass_at_k = {}
#     for k in k_list:
#         pass_at_k[k] = cnt/len(code_contests)
#         print(f'pass@{k} (kulal): {round(pass_at_k[k]*100, 2)}%')
        
#     return pass_at_k, pass_at_k


# -

def get_code_contests_prompt(description, tokenizer, margin=15):
#     input_data = f"\nQUESTION:\n{}\nUse Standard Input format\nANSWER:\n"    
    _input = "\nQUESTION:\n"
    _input += tokenizer.decode(tokenizer.encode(description)[:tokenizer.model_max_length-margin])
    _input += "\nUse Standard Input format"        
    _input += "\nANSWER:\n"
    
    return _input


# +
import json
import argparse

from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def evaluate_competition(dataset, candidates_list, k_list=[1], dataset_name='apps'):
    timeout = 4  # seconds
    passed = {}
    for i in tqdm(range(len(dataset))):
        if i == len(candidates_list):
            break
        d = dataset[i]
        if dataset_name == 'apps':
            tests = json.loads(d['input_output'])
        elif dataset_name == 'code_contests':
            tests = get_tests(d)
        passed_list = [None] * len(tests['inputs'])
        
        if len(candidates_list[i]) != 1 or k_list != [1]:
            raise NotImplementedError
            
        for n, sol in enumerate(candidates_list[i]):
            code_fn = f"../apps/preds/{i}/sol_{n}.py"
            os.makedirs("/".join(code_fn.split("/")[:-1]), exist_ok=True)
            flag = True
            with open(code_fn, 'w', encoding='utf-8') as fp:
                fp.write(sol)
            for j in range(len(tests['inputs'])):
                tests['inputs'][j] = str(tests['inputs'][j])
                tests['outputs'][j] = str(tests['outputs'][j])
                input_fn = f"../{dataset_name}/inputs/{i}/input_{j}.txt"
                output_fn = f"../{dataset_name}/outputs/{i}/output_{j}.txt"
                os.makedirs("/".join(input_fn.split("/")[:-1]), exist_ok=True)
                os.makedirs("/".join(output_fn.split("/")[:-1]), exist_ok=True)
                with open(input_fn, 'w', encoding='utf-8') as fp:
                    fp.write(tests['inputs'][j])
                os.system(f"timeout {timeout}s python {code_fn} < {input_fn} > {output_fn}")
                with open(output_fn, 'r', encoding='utf-8') as fp:
                    stdout_data = fp.read()
                is_passed = stdout_data.lstrip().rstrip() == tests['outputs'][j].lstrip().rstrip()
                passed_list[j] = (is_passed, tests['inputs'][j], tests['outputs'][j], stdout_data)
                if not is_passed:
                    flag = False
                    break
            passed[i] = passed_list
            if flag:
                print(f"[PASSED] INDEX {i}: {n}th solution")
    
    # Count the number of all-passed predictions
    cnt = 0
    for item in passed.values():
        flag = True
        for v in item:
            if not v[0]:
                flag = False
                break
        if flag:
            cnt += 1
            
    pass_at_k = {}
    for k in k_list:
        pass_at_k[k] = cnt/len(candidates_list)
        print(f'pass@{k} (kulal): {round(pass_at_k[k]*100, 2)}%')
        
    return pass_at_k, pass_at_k

# +
# answer_type = "\nUse Standard Input format"
# if json.loads(d["input_output"]).get("fn_name"):
#     answer_type = "\nUse Call-Based format"
# footer = "\n" + answer_type + "\nANSWER:"+"\n"
# init_footer_len = len(tokenizer.tokenize(footer))
# if d["starter_code"] !='':
#     footer += d["starter_code"] + "\n"
# truncate_len = len(tokenizer.tokenize(footer))
# if truncate_len >= tokenizer.model_max_length:
#     footer = tokenizer.convert_tokens_to_string(tokenizer.tokenize(footer)[:-cmd_args.max_new_tokens])
# if 'p-PUBLIC' in cmd_args.model_name:
#     truncate_len += 1
# _input_data = "#QUESTION:\n" + d["question"]
# # Truncate question if overflows
# tokenized_input_data = tokenizer.tokenize(_input_data)[:tokenizer.model_max_length]
# diff_to_max_len = len(tokenized_input_data) + truncate_len + cmd_args.max_new_tokens - tokenizer.model_max_length
# if diff_to_max_len > 0:
#     _input_data = tokenizer.convert_tokens_to_string(tokenized_input_data[:max(600-init_footer_len,len(tokenized_input_data)-diff_to_max_len)])
# else:
#     _input_data = tokenizer.convert_tokens_to_string(tokenized_input_data[:600-init_footer_len])

# _input_data += footer
# input_data = copy.deepcopy(_input_data)
# if 'p-PUBLIC' in cmd_args.model_name:
#     input_data += '<|python|>'
# print("="*100)
# print(input_data)

# +
# from apps.eval.generate_gpt_codes import reindent_code

# def generate_prompt(args, d):
#     _input = "\nQUESTION:\n"
#     _input += d['questions']
#     if starter_path != None:
#         with open(starter_path, "r") as f:
#             data = f.readlines()
#             data = "".join(data)
#             data = "\n" + data #+ "\n"
#         _input += data
#     else:
#         #_input += "\n\n"
#         pass

#     with open(test_case_path, "r") as f:
#         data = json.load(f)
#     if not data.get("fn_name"):
#         _input += "\nUse Standard Input format"#\n"
#     else:
#         _input += "\nUse Call-Based format"#\n"
    
#     _input += "\nANSWER:\n"

#     if args.peeking > 0.0:
#         # Need to do some peeking. 

#         # Read one example solution
#         with open(solutions_path, 'r') as f:
#             sols = json.load(f)

#         # Choose the shortest solution for the model to use.
#         # This is so we can conserve tokens (1024 max)
#         # sample_sol = min(sols, key=len)

#         # # Add args.peeking% of that solution to the prompt
#         # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
#         # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
#         # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
#         # _input += tokenizer.decode(sample_sol_token_ids)

#         # Alternatively take a random solution
#         sample_sol = random.choice(sols)
#         rand_sol = reindent_code(sample_sol)
#         rand_sol = tokenizer.encode(rand_sol, verbose=False)
#         tokens_taken = int(args.peek_frac * len(rand_sol))
#         rand_sol = rand_sol[:tokens_taken]
#         _input += tokenizer.decode(rand_sol)
#     else:
#         sample_sol = None

#     return _input, sample_sol
