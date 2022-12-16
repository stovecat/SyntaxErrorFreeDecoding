# -*- coding: utf-8 -*-
import os
import copy
import json
import torch
import pickle
import argparse
from utils import *
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from subprocess import Popen, PIPE, STDOUT

################################################################################################################
# Parser
################################################################################################################
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="human_eval",
                        help='Evaluation dataset name.')
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5-large-ntp-py',
                        help='Code generation model name.')
    parser.add_argument('--decoding', type=str, default='nucleus',
                        help='Decoding method.')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p for nucleus sampling.')    
    parser.add_argument('--T', type=float, default=0.0,
                        help='Temperature for nucleus sampling and tempering.')    
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum number of tokens to generate.')   
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help='The number of samples to generate for each input.')   
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')   
    parser.add_argument('--time', type=str, default="", help='log_time')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--apps_level', type=int, default=-1)
    parser.add_argument('--apps_max_description_tokens', type=int, default=600)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--dataset_type', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stop_token', type=str, default='eos')
    parser.add_argument('--syntax_error_free', type=str, default="false")
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--curriculum_level', type=int, default=512)
    parser.add_argument('--DEBUG', action='store_true')

    return parser.parse_args()


################################################################################################################
# Load dataset
################################################################################################################
def load_code_dataset(cmd_args):
    if cmd_args.dataset_name == "human_eval":
        human_eval = load_dataset("openai_humaneval")
        dataset=human_eval['test']
    elif cmd_args.dataset_name == "code_contests":
        raise NotImplementedError
    elif cmd_args.dataset_name == "apps":
    #     dataset = load_dataset("codeparrot/apps", split=cmd_args.dataset_type)#, difficulties=["introductory"])
        dataset = load_pkl(f"./apps/{cmd_args.dataset_type}_level_{cmd_args.curriculum_level}.pkl")
    else:
        raise NotImplementedError
    return dataset


################################################################################################################
# Preprocess input data
################################################################################################################
def get_input_data(d, cmd_args, tokenizer, count=0, log=True):
    if log:
        print("="*100)
        print(f"INDEX: {count-1}")
    if cmd_args.dataset_name == "human_eval":
        _input_data = d['prompt']
        input_data = copy.deepcopy(_input_data)
        if 'p-PUBLIC' in cmd_args.model_name:
            input_data = '<|python|>'+input_data
        if log:
            print("="*100)
    elif cmd_args.dataset_name == "code_contests":
        max_len = cmd_args.max_length
        if 'p-PUBLIC' in cmd_args.model_name:
            max_len -= 1
        _input_data = get_1_shot_input(train, d, tokenizer, max_input_length=max_len, lang='python')
        input_data = copy.deepcopy(_input_data)
        if 'p-PUBLIC' in cmd_args.model_name:
            input_data += '<|python|>'
        if log:
            print(f"INPUT DATA")
            print("="*100)
            print(input_data)
            print()
            print(cmd_args.max_length)
    elif cmd_args.dataset_name == "apps":
        answer_type = "\nUse Standard Input format"
#         print(f'type(d["input_output"])\n{type(d["input_output"])}')
        try:
            if d["input_output"] != '' and json.loads(d["input_output"]).get("fn_name"):
                answer_type = "\nUse Call-Based format"
        except Exception as e:
            print(e)
            print(f"INDEX: {count}")
#             print(d["input_output"])
            assert 1==2
        footer = "\n" + answer_type + "\nANSWER:\n"
        init_footer_len = len(tokenizer.tokenize(footer))
        if d["starter_code"] !='' and check_syntax_error(d["starter_code"], has_eos=False) is None:
            footer += d["starter_code"] + "\n"
        truncate_len = len(tokenizer.tokenize(footer))
        if truncate_len >= cmd_args.max_length:
            footer = tokenizer.convert_tokens_to_string(tokenizer.tokenize(footer)[:-cmd_args.max_new_tokens])
        if 'p-PUBLIC' in cmd_args.model_name:
            truncate_len += 1
        _input_data = "\nQUESTION:\n" + d["question"]
        # Truncate question if overflows
        tokenized_input_data = tokenizer.tokenize(_input_data)[:cmd_args.max_length]
        diff_to_max_len = len(tokenized_input_data) + truncate_len + cmd_args.max_new_tokens - cmd_args.max_length
        if diff_to_max_len > 0:
            _input_data = tokenizer.convert_tokens_to_string(tokenized_input_data[:max(cmd_args.apps_max_description_tokens-init_footer_len,len(tokenized_input_data)-diff_to_max_len)])
        else:
            _input_data = tokenizer.convert_tokens_to_string(tokenized_input_data[:cmd_args.apps_max_description_tokens-init_footer_len])

        _input_data += footer
        input_data = copy.deepcopy(_input_data)
        if 'p-PUBLIC' in cmd_args.model_name:
            input_data += '<|python|>'
        if log:
            print("="*100)
            print(input_data)
    return input_data, _input_data

    
################################################################################################################
# Save generated codes
################################################################################################################
def get_save_path(cmd_args):
    name_path = cmd_args.model_name
    name_path = name_path.replace('/', '_')
    fn = f'{cmd_args.time}_{cmd_args.dataset_name}_{name_path}_{cmd_args.decoding}'
    if cmd_args.decoding == 'greedy':
        pass
    elif cmd_args.decoding == 'beam':
        fn +=  f'_{cmd_args.num_return_sequences}'
    elif cmd_args.decoding == 'nucleus':
        fn +=  f'_p={cmd_args.top_p}_T={cmd_args.T}'
    elif cmd_args.decoding == 'tempering':
        fn +=  f'_T={cmd_args.T}'
    else:
        raise NotImplementedError
    os.makedirs(f"./results/{fn}", exist_ok=True)
    os.makedirs(f"./eval/{fn}", exist_ok=True)
    
    return fn
    
    
def dump_candidate_list(candidates_list, cmd_args):
    fn = get_save_path(cmd_args)
    fn = f"./results/{fn}"
    # with open(fn+f'_rank_{cmd_args.local_rank}.pkl', 'wb') as fp:
    if cmd_args.end_idx > 0:#cmd_args.start_idx !=0 and cmd_args.end_idx != len(dataset):
        fn += f"/{cmd_args.start_idx}-{cmd_args.end_idx}"
    else:
        fn += "/all"
    with open(fn+f'.pkl', 'wb') as fp:
        pickle.dump(candidates_list, fp)

        
################################################################################################################
# Test Run
################################################################################################################
def _test_run(sol, tests, idx, n, dataset_name="APPS", timeout=4, passed=None, dataset_type='test', time_stamp=""):
    code_fn = f"_test_run/preds/{idx}/sol_{time_stamp}_{n}.py"
    os.makedirs("/".join(code_fn.split("/")[:-1]), exist_ok=True)
    flag = 0
    import_list = ["math", "sys", "collections", "functools", "itertools", "heapq", "random", "copy"]
    sol = "".join([f"import {v}\nfrom {v} import *\n" for v in import_list])+"del globals()['pow']\n"+sol
    if tests.get("fn_name"):
        if "class Solution" in sol:
            sol = f"from typing import *\n{sol}\nimport json\nsol = Solution()\nv = json.loads(input())\nprint(sol.{tests['fn_name']}(*v))"
        else:
            sol = f"from typing import *\n{sol}\nimport json\nv = json.loads(input())\nprint({tests['fn_name']}(*v))"
    with open(code_fn, 'w', encoding='utf-8') as fp:
        fp.write(sol)
        
    # Errors in the original test cases
    if dataset_type=='train':
        if idx == 358: 
            tests = {
                        "fn_name": "findReplaceString",
                        "inputs": [["'abcd'", [0, 2], ["'a'", "'cd'"], ["'eee'", "'ffff'"]], 
                                   ["'abcd'", [0, 2], ["'ab'", "'ec'"], ["'eee'", "'ffff'"]]],
                        "outputs": ["'eeebffff'", "'eeecd'"]
                    }
        elif idx == 1673:
            tests["inputs"] = [[[1, 2, 3], [6, 6, 7], [7, 8, 9], [], []]]
        elif idx == 1674:
            tests["inputs"] = [[2,7,9,4,4]]
        elif idx == 1677:
            tests["inputs"] = ['52\n3 2 3 13\n 4 4 6 3']
        elif idx == 2402:
            tests["inputs"] = ["\'Let\'s take LeetCode contest\'"]
        elif idx == 2642:
            tests["inputs"] = ["HackerRank.com presents 'Pythonist 2'."]
        elif idx == 2884:
            tests['inputs'] = ["None"]
        elif idx == 2971:
            tests = {
                        "fn_name": "watch_pyramid_from_the_side",
                        "inputs": ["abc", ""],
                        "outputs": ["  c  \n bbb \naaaaa", ""]
                }

    passed_list = [None] * len(tests['inputs'])
    for j in range(len(tests['inputs'])):
        if dataset_type=='train':
            # Handling wrong input formats
            if (idx in range(514, 1603) or 
                idx in [1670, 1675, 2328, 2329, 2349, 2350, 2351] or 
                idx in range(1676,1692)) and type(tests['inputs'][j]) is list:
                tests['inputs'][j] = "\n".join([str(v) for v in tests['inputs'][j]])
            if idx == 2662:
                tests['inputs'][j] = tests['inputs'][j][0]
            if idx == 2676:
                tests['inputs'][j] = [[str(v) for v in tests['inputs'][j][0]]]
            if idx == 3094:
                tests['inputs'][j] = tests['inputs'][j][0]
                tests['outputs'][j] = tests['outputs'][j][0]
            if idx == 3146:
                tests['inputs'][j] = tests['inputs'][j][0]
            if idx == 2656:
                tests['inputs'][j] = tests['inputs'][j][0]
            tests['inputs'][j] = str(tests['inputs'][j])

            if tests['inputs'][j] == 'None':
                tests['inputs'][j] = '"None"'
        else:
            tests['inputs'][j] = str(tests['inputs'][j])
            
        outputs = str(tests['outputs'][j])
        # Handling JsonDecoderError
        if "'" in tests['inputs'][j] and '"' not in tests['inputs'][j]:
            tests['inputs'][j] = tests['inputs'][j].replace("'", '"')
        import re
        p = re.compile("(\[|(.*,))\s*'\".*\"'")
        m = p.match(tests['inputs'][j])
        if m is not None:
            def swap_words(s, x, y):
                return y.join(part.replace(y, x) for part in s.split(x))
            tests['inputs'][j] = swap_words(tests['inputs'][j], "'\"", "\"'")
            
        input_fn = f"_test_run/inputs/{idx}/input_{time_stamp}_{j}.txt"
        output_fn = f"_test_run/outputs/{idx}/output_{time_stamp}_{j}.txt"
        os.makedirs("/".join(input_fn.split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(output_fn.split("/")[:-1]), exist_ok=True)
        with open(input_fn, 'w', encoding='utf-8') as fp:
            fp.write(tests['inputs'][j])
        cmd = f"cd _test_run\ntimeout {timeout}s python {code_fn.replace('_test_run/', '')} < {input_fn.replace('_test_run/', '')} > {output_fn.replace('_test_run/', '')}\ncd .."
        os.system(cmd)
        with open(output_fn, 'r', encoding='utf-8') as fp:
            stdout_data = fp.read()
        is_passed = stdout_data.lstrip().rstrip() == outputs.lstrip().rstrip()
        if type(tests['outputs'][j]) is list:
            if stdout_data.lstrip().rstrip() == "\n".join([str(v) for v in tests['outputs'][j]]).lstrip().rstrip():
#                 print('it was list')
                is_passed = True
        
        passed_list[j] = (is_passed, tests['inputs'][j], tests['outputs'][j], stdout_data)
        if not is_passed:
            break
        flag += 1
    if passed is not None:
        passed[idx].append(passed_list)
    if flag == len(tests['inputs']):
        flag = 1
    else:
        flag /= len(tests['inputs'])
    return passed, flag

def evaluate_competition(dataset, start_idx, candidates_list, dataset_name='apps', dataset_type='test', time_stamp=""):
    timeout = 4  # seconds
    log = {}
    passed = {}
    for i in tqdm(range(len(dataset))):
        log[i] = []
        if i == len(candidates_list):
            break
        d = dataset[start_idx+i]
        if dataset_name == 'apps':
            if d['input_output'] == '':
                # No test cases
                passed[i+start_idx] = [[(False, None, None, None)] for _ in candidates_list[i]]
                continue
            tests = json.loads(d['input_output'])
            if tests['inputs'] == [] and tests['outputs'] == []:
                passed[i+start_idx] = [[(False, None, None, None)] for _ in candidates_list[i]]
                continue                
        elif dataset_name == 'code_contests':
            tests = get_tests(d)
            
        passed[i+start_idx] = []
        for n, sol in enumerate(candidates_list[i]):
            if dataset_name == 'apps':
                tests = json.loads(d['input_output'])
            passed, flag = _test_run(sol, tests, i+start_idx, n, dataset_name=dataset_name, passed=passed, dataset_type=dataset_type, time_stamp=time_stamp)  
            log[i].append((n, flag))
            if flag == 1:
                print(f"[PASSED] INDEX {start_idx+i}: {n}th solution")
            
    # Count the number of all-passed predictions
    cnt_dict = {diff: [] for diff in ['interview', 'competition', 'introductory', 'all']}
    for idx, items in passed.items():
        difficulty = dataset[idx]['difficulty']
        cnt_dict[difficulty].append([])
        cnt_dict['all'].append([])
        for item in items:
            flag = True
            for v in item:
                if v is None:
                    print(f"NONE ERROR IN PROB IDX {i+start_idx}, SOL IDX {idx}")
                    flag = False
                    break
                if not v[0]:
                    flag = False
                    break
            if flag:
                cnt_dict[difficulty][-1].append(1)
                cnt_dict['all'][-1].append(1)
            else:
                cnt_dict[difficulty][-1].append(0)
                cnt_dict['all'][-1].append(0)
    
    return cnt_dict, passed
        

    
################################################################################################################
# Main 
################################################################################################################
if __name__ == "__main__":
    cmd_args = get_parser()
    if cmd_args.syntax_error_free == 'true':
        cmd_args.syntax_error_free = True
    else:
        cmd_args.syntax_error_free = False
    set_seed(cmd_args.seed, 16) 

    model, tokenizer = get_model(cmd_args.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if cmd_args.stop_token == 'eos' and not cmd_args.syntax_error_free:
        stop_id = tokenizer.eos_token_id
    elif cmd_args.stop_token == 'new_line' or cmd_args.syntax_error_free:
        stop_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("\n")[0])
    else:
        raise NotImplementedError
    stop_ids = tokenizer("\n\n\n<|endoftext|>", add_special_tokens=False).input_ids
    stopping_criteria = StoppingCriteriaList(stop_ids)

    model.to(cmd_args.gpu)
    dataset = load_code_dataset(cmd_args)
    if cmd_args.end_idx == -1:
        cmd_args.end_idx = len(dataset)
    cmd_args.max_length = min(cmd_args.max_length, tokenizer.model_max_length)


    inner_n_samples = 1
    if cmd_args.num_return_sequences > 1:
        if cmd_args.num_return_sequences < 10:
            inner_n_samples = cmd_args.num_return_sequences
        elif cmd_args.num_return_sequences % 10 == 0:
            inner_n_samples = 10

    print(cmd_args)
    
    candidates_list = []
    count = 0
    for d in tqdm(dataset):
        if not (cmd_args.start_idx <= count < cmd_args.end_idx):
            count += 1
            continue
        count += 1

        input_data, _input_data = get_input_data(d, cmd_args, tokenizer, count)

        print("="*100)
        n_samples = []
        input_data_dict = None
        # for syntax error free decoding
        prev_token_len = 0
        status = [] # done / no_error / error
        _inner_n_samples = None
        next_inputs = []
        prev_inputs = []
        init_token_len = None
        patience_status = []
        next_patience_status = []
        with torch.no_grad():
            while len(n_samples) < cmd_args.num_return_sequences:
                ################################################################################################################
                # Initialize input
                ################################################################################################################
                if not cmd_args.stop_token == 'new_line' or \
                    (cmd_args.stop_token == 'new_line' and \
                         (len(prev_inputs) == 0 or len(set(status)) == 1 and 'done' in status)):
                    if cmd_args.DEBUG:
                        print("input_data", flush=True)
                    input_data_dict = tokenizer(input_data, return_tensors="pt", 
                                                padding=True, truncation=True).to(model.device)
                    init_token_len = input_data_dict['input_ids'].size()[1]
                    _inner_n_samples = inner_n_samples
                    patience_status = [cmd_args.patience]*_inner_n_samples
                else:
                    del input_data_dict
                    if cmd_args.DEBUG:
                        print("next_inputs", flush=True)
                    next_inputs_t = [tokenizer(ni, return_tensors="pt", truncation=True)['input_ids'][0].flip(0) for ni in next_inputs]
                    input_ids = pad_sequence(next_inputs_t, padding_value=tokenizer.pad_token_id, batch_first=True).flip(1)
                    attention_mask = (tokenizer.pad_token_id != input_ids)
                    position_ids = attention_mask.cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask.eq(0), 0)
                    input_data_dict = {
                        'input_ids': input_ids.to(model.device),
                        'attention_mask': attention_mask.to(model.device),
                        'position_ids': position_ids.to(model.device)
                    }
                    _inner_n_samples = 1

                max_new_tokens = max(0, cmd_args.max_length - input_data_dict['input_ids'].size()[-1])
                max_new_tokens = min(max_new_tokens, cmd_args.max_new_tokens)

                if cmd_args.stop_token == 'new_line':
                    max_new_tokens = max(max_new_tokens, cmd_args.max_length-prev_token_len-init_token_len)

                if cmd_args.DEBUG:
                    print(cmd_args.max_length, input_data_dict['input_ids'].size()[-1], max_new_tokens, cmd_args.max_new_tokens, flush=True)

                torch.cuda.empty_cache()
                ################################################################################################################
                # Decoding
                ################################################################################################################
                # greedy-decoding
                if cmd_args.decoding == 'greedy':
                    generated_ids = model.generate(**input_data_dict, max_new_tokens=max_new_tokens, 
                                                   early_stopping=True, pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=stop_id, stopping_criteria=stopping_criteria)
                # beam
                elif cmd_args.decoding == 'beam':
                    generated_ids = model.generate(**input_data_dict, max_new_tokens=max_new_tokens, num_beams=inner_n_samples,
                                                   num_return_sequences=_inner_n_samples, 
                                                   early_stopping=True, pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=stop_id, stopping_criteria=stopping_criteria)
                # nucleus sampling
                elif cmd_args.decoding == 'nucleus':
                    generated_ids = model.generate(**input_data_dict, max_new_tokens=max_new_tokens, do_sample=True, 
                                                   top_p=cmd_args.top_p, temperature=cmd_args.T, early_stopping=True, 
                                                   num_return_sequences=_inner_n_samples, 
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=stop_id, stopping_criteria=stopping_criteria)
                # tempering
                elif cmd_args.decoding == 'tempering':
                    generated_ids = model.generate(**input_data_dict, max_new_tokens=max_new_tokens, do_sample=True, 
                                                   temperature=cmd_args.T, early_stopping=True, 
                                                   num_return_sequences=_inner_n_samples, 
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=stop_id, stopping_criteria=stopping_criteria)
                else:
                    raise NotImplementedError


                ################################################################################################################
                # Post process: Syntax-error-free
                ################################################################################################################
                if cmd_args.stop_token == 'new_line':
                    prev_token_len = generated_ids.size()[1] - init_token_len 
                    # Step 3. Check syntax errors
                    current_batch_size = generated_ids.size()[0]
                    current_outputs = []
                    current_truncated_output = []
                    status = []
                    next_inputs = []
                    for gen_idx in range(current_batch_size):
                        has_eos = False
                        if len(n_samples) == cmd_args.num_return_sequences:
                            break
                        gen = generated_ids[gen_idx]
                        if tokenizer.eos_token_id in gen:
                            has_eos = True
                            eos_idx = get_index_by_value(gen, tokenizer.eos_token_id).item()
                            gen = gen[:eos_idx]

                        output = tokenizer.decode(gen, skip_special_tokens=True)
                        del gen
                #         print(output)
                        current_outputs.append(output)

                        split_output = output.split("\nANSWER:\n")
                        truncated_output = truncate_before_pattern("\nANSWER:\n".join(split_output[1:]), 
                                                               ["<|python|>", "<|", "<|/", "<code>", "</code>", "<cell>", "</cell>", "<text>", "</text>"])
                        current_truncated_output.append(truncated_output)

                        if has_eos or prev_token_len >= cmd_args.max_new_tokens or max_new_tokens == 0:
                            if not cmd_args.syntax_error_free:
                                n_samples.append(truncated_output)
                                status.append('done')
                            elif check_syntax_error(truncated_output, has_eos=True) is None:
                                n_samples.append(truncated_output)
                                status.append('done')
                            else:
                                status.append('error')
                                patience_status[gen_idx] -= 1
                                print("-"*40)
                                print(f"[gen_idx: {gen_idx}]")
                                print(truncated_output)
                                print("-"*40)
                        else:
                            if not cmd_args.syntax_error_free:
                                status.append('continue')
                            elif check_syntax_error(truncated_output) is None:
                                status.append('no_error')
                            else:
                                status.append('error')
                                patience_status[gen_idx] -= 1
                                print("-"*40)
                                print(f"[gen_idx: {gen_idx}]")
                                print(truncated_output)
                                print("-"*40)
                    if len(n_samples) == cmd_args.num_return_sequences:
                        break

                    assert len(status) == current_batch_size
                    print(current_batch_size, status)

                    # Step 4. consist next inputs
                    for gen_idx in range(current_batch_size):
                        if cmd_args.DEBUG:
                            print(f"[gen_idx: {gen_idx}]")
                        if status[gen_idx] == 'done':
                            if cmd_args.DEBUG:
                                print("done")
                            pass
                        elif not cmd_args.syntax_error_free:
                            if cmd_args.DEBUG:
                                print("No syntax filtering")
                            next_inputs.append(current_outputs[gen_idx])
                        elif (patience_status[gen_idx] > 0) and (status[gen_idx] in ['error', 'comment']):
                            if len(prev_inputs) == 0:
                                if cmd_args.DEBUG:
                                    print(f"error when generating first line (patience {patience_status[gen_idx]})")
                                next_inputs.append(input_data)
                                next_patience_status.append(cmd_args.patience)
                            else:
                                if cmd_args.DEBUG:
                                    print(f"error when generating line > 1 (patience {patience_status[gen_idx]})")
                                next_inputs.append(prev_inputs[gen_idx])
                                next_patience_status.append(patience_status[gen_idx])
                        elif status[gen_idx] == 'error':
                            # select successed other input 
                            if len(prev_inputs) == 0:
                                if cmd_args.DEBUG:
                                    print(f"[!] error when generating first line (patience {patience_status[gen_idx]})")
                                next_inputs.append(input_data)
                                next_patience_status.append(cmd_args.patience)
                            else:
                                if cmd_args.DEBUG:
                                    print(f"[!] error when generating line > 1 (patience {patience_status[gen_idx]})")
                                if 'no_error' in status or 'done' in status:
                                    selected_idx = random.choice([_idx for _idx, _s in enumerate(status) if _s in ['no_error', 'done']])
                                    if cmd_args.DEBUG:
                                        print(f"select other idx: {selected_idx}")
                                    next_inputs.append(prev_inputs[selected_idx])
                                    next_patience_status.append(patience_status[selected_idx])
                                else:
                                    if cmd_args.DEBUG:
                                        print("EVERY STATUS IS ERRONEOUS")
                                    n_samples.append(current_truncated_output[gen_idx]+tokenizer.eos_token)
                        else:
                            if cmd_args.DEBUG:
                                print("So far so good!")
                            next_inputs.append(current_outputs[gen_idx])   
                            next_patience_status.append(patience_status[gen_idx])
                    if cmd_args.DEBUG:
                        print(f"Next batch size: {len(next_inputs)}")
                    assert len(next_inputs) <= inner_n_samples
                    prev_inputs = next_inputs
                    patience_status = next_patience_status
                ################################################################################################################
                # Post process: Default
                ################################################################################################################
                else:
                    for gen_idx in range(inner_n_samples):
                        gen = generated_ids[gen_idx]
                        if 'Salesforce/codegen' in cmd_args.model_name:
                            output = tokenizer.decode(gen, skip_special_tokens=True)
                        elif 'facebook/incoder' in cmd_args.model_name:
                            output = tokenizer.decode(gen, 
                                                      clean_up_tokenization_spaces=False, skip_special_tokens=True)
                        else:
                            output = tokenizer.decode(gen, skip_special_tokens=True)

                        if cmd_args.dataset_name == "apps":
                            split_output = output.split("\nANSWER:\n")
                            truncated_output = truncate_before_pattern("\nANSWER:\n".join(split_output[1:]), 
                                                                   ["<|python|>", "<|", "<|/", "<code>", 
                                                                    "</code>", "<cell>", "</cell>", "<text>", "</text>"])
                            n_samples.append(truncated_output)
                        #T5계열은 input output concat 필요    
                        elif 'codet5' in cmd_args.model_name:
                            truncated_output = truncate_before_pattern(output, 
                                                                   ["\nclass", "\ndef", "<|", "<|/", "<code>", "</code>", 
                                                                    "<cell>", "</cell>", "<text>", "</text>", r"\n\n^#", "^'''", "\n\n\n"])  
                            if cmd_args.dataset_name == "human_eval":
                                n_samples.append(_input_data+truncated_output)
                            elif cmd_args.dataset_name in ["code_contests", "apps"]:
                                n_samples.append(truncated_output)
                            else:
                                raise NotImplementedError
                        # GPT 계열은 그냥 output만
                        else:
                            split_output = output.split(_input_data)
                            if len(split_output) == 1:
                                truncated_output = truncate_before_pattern(output, 
                                                                       ["<|", "<|/", "<code>", "</code>", 
                                                                        "<cell>", "</cell>", "<text>", "</text>"])            
                                n_samples.append(truncated_output)
                            else:
                                truncated_output = truncate_before_pattern(split_output[1], 
                                                                       ["\nclass", "\ndef", "<|", "<|/", "<code>", "</code>", 
                                                                        "<cell>", "</cell>", "<text>", "</text>", r"\n\n^#", "^'''", "\n\n\n"])
                                if cmd_args.dataset_name == "human_eval":
                                    n_samples.append(_input_data+truncated_output)
                                elif cmd_args.dataset_name == "code_contests":
                                    n_samples.append(truncated_output)
                                else:
                                    raise NotImplementedError

                if cmd_args.decoding in ['greedy', 'beam']:
                    break
            candidates_list.append(n_samples)
            print("="*100)
            print(f"OUTPUT")
            print("="*100)
            for cand in n_samples:
                print(cand)
                print("="*100,'\n', flush=True)

        if cmd_args.DEBUG and count > cmd_args.start_idx+10:
            break

            
    ################################################################################################################
    # Save generated codes
    ################################################################################################################    
    dump_candidate_list(candidates_list, cmd_args)
    
    ################################################################################################################
    # Evaluate
    ################################################################################################################    
    cnt_dict, passed = evaluate_competition(dataset, cmd_args.start_idx, candidates_list, 
                                            dataset_name=cmd_args.dataset_name, 
                                            dataset_type=cmd_args.dataset_type, time_stamp=cmd_args.time)
    if cmd_args.DEBUG:
        print(cnt_dict)
        
    fn = get_save_path(cmd_args)
    dump_pkl(f'./eval/{fn}/cnt_dict_{cmd_args.start_idx}-{cmd_args.end_idx}.pkl', cnt_dict)
    dump_pkl(f'./eval/{fn}/passed_{cmd_args.start_idx}-{cmd_args.end_idx}.pkl', passed)
