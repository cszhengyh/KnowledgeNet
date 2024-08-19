import os
import re
import sys
import json
import jsonlines
sys.path.append("/share/project/zhengyuhui/domain_tree_3.1.1")
from chat_with_gpt import chat_with_gpt
from extract_answer import *
from prompts.check_answer_prompt import check_answer_prompt
from collections import defaultdict

tot_instruction = 0
# science_pres = ['agieval', 'ARC-c', 'lukaemon_mmlu', 'Gaokao', 'cmmlu', 'ceval', 'GPQA']
# science_pres = ['ARC-c', 'lukaemon_mmlu', 'GPQA'] # en
science_pres = ['ceval'] # zh

def extract_bad_case_from_openai_humaneval(human_eval_instructions_path):
    out = ""
    with jsonlines.open(HUMAN_EVAL_RESULT_PATH, 'r') as HUMAN_EVAL:
        for line in HUMAN_EVAL:
            global tot_instruction
            tot_instruction += 1
            if line['passed'] == False:
                with open(os.path.join(INSTRUCTIONS_DIRNAME, human_eval_instructions_path), 'r', encoding='utf-8') as HUMAN_EVAL_INSTRUCTIONS:
                    instructions = json.load(HUMAN_EVAL_INSTRUCTIONS)
                    for item in instructions.values():
                        if line['task_id'] == item['gold']:
                            with jsonlines.open(HUMAN_EVAL_DATA_PATH, 'r') as human_eval_data:
                                for data_line in human_eval_data:
                                    if data_line['task_id'] == line['task_id']:
                                        out += repr(item['origin_prompt']+data_line['canonical_solution']).strip('"\'')+'\n'
                                        break
                            break

    with open(f"{BAD_CASE_INSTRUCTIONS_DIRNAME}/openai_humaneval_bad_case.instructions", 'w', encoding='utf-8') as outfile:
        outfile.write(out)

def check_prediction(f, query, prediction, gold):
    # return True

    # regulation
    answer = globals()[f"extract_{f.replace('-', '_')}_answer"](prediction)
    return answer != gold

    # gpt4o
    # query = check_answer_prompt + f"The question is \"{query}\", the student's answer P is \"{prediction}\" and the standard correct answer is \"{gold}\"."    
    # while True:
    #     check = chat_with_gpt(query)
    #     if re.findall(r'\[result: \s*(.*?)\]', check):
    #         check = re.findall(r'\[result: \s*(.*?)\]', check)[0].lower()
    #         with open(f"{LOG_PATH}/{f}.log", 'a', encoding='utf-8') as output_file:
    #             output_file.write(rf"gpt4o -> {check}"+'\n')
    #             output_file.write(rf"prediction -> {prediction}"+'\n')
    #             output_file.write(rf"gold -> {gold}"+'\n')            
    #             output_file.write("\n=============================\n")            
    #         if check == 'wrong':
    #             return True
    #         return False

def extract_bad_case_from_mbpp(f):
    out = ""
    with open(f"{INSTRUCTIONS_DIRNAME}/{f}", 'r', encoding='utf-8') as mbpp_instructions:
        instructions = json.load(mbpp_instructions)
        for item in instructions.values():
            out += repr(item['origin_prompt']).strip('"\'')+'\n'
    with open(f"{BAD_CASE_INSTRUCTIONS_DIRNAME}/mbpp_bad_case.instructions", 'w', encoding='utf-8') as outfile:
        outfile.write(out)

for f in os.listdir(INSTRUCTIONS_DIRNAME):
    outputs = defaultdict(lambda: defaultdict(lambda: -1))
    if f.endswith(".json"):
        # if f[:f.find('.')] != 'GPQA_diamond':
        #     continue

        flag = 0
        check_prediction_func_name = ""
        for science_pre in science_pres:
            if f[:len(science_pre)] == science_pre:
                flag = 1
                check_prediction_func_name = science_pre
                break
        if flag == 0:
            continue

        if f[:f.find('.')] == 'openai_humaneval':
            extract_bad_case_from_openai_humaneval(f)
            continue

        if f[:f.find('.')] == 'mbpp':
            extract_bad_case_from_mbpp(f)
            continue

        pre = f[:f.find('.')]
        with open(os.path.join(INSTRUCTIONS_DIRNAME, f), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            for item in data.values():
                try:
                    item['gold']
                except Exception as e:
                    continue
                if 'origin_prompt' in item.keys():
                    if isinstance(item['origin_prompt'], str):
                        query = item['origin_prompt']
                        if isinstance(item['gold'], str):
                            if check_prediction(check_prediction_func_name, query, item['prediction'], item['gold']):
                                outputs[pre][query] = item['gold']
                        elif isinstance(item['gold'], dict):
                            best_ans_pre, corr_ans_pre, incorr_ans_pre = "The best answer is \"", "the correct answers are \"", "the incorrect answers are \""
                            best_ans, corr_ans, incorr_ans = "The best answer is \"", "the correct answers are \"", "the incorrect answers are \""
                            best_ans += item['gold']['answers']['best_answer']+"\", "
                            for subitem in item['gold']['answers']['correct_answers']:
                                corr_ans += subitem+"\", \""
                            for subitem in item['gold']['answers']['incorrect_answers']:
                                incorr_ans += subitem+"\", \""
                            answer = best_ans + corr_ans[:-1] + incorr_ans[:-3]
                            if check_prediction(check_prediction_func_name, query, item['prediction'], best_ans[len(best_ans_pre)-1:]+corr_ans[len(corr_ans_pre)-1:-3]):
                                outputs[pre][query] = answer
                        else:
                            ans, sep = "", ", "
                            for subitem in item['gold']:
                                ans += f"\"{subitem}\"{sep}"
                            if check_prediction(check_prediction_func_name, query, item['prediction'], ans[:-len(sep)]):                                
                                outputs[pre][query] = ans[:-len(sep)]
                    else:
                        query = ""
                        for subitem in item['origin_prompt']:
                            query += subitem['content']
                        if check_prediction(check_prediction_func_name, query, item['prediction'], item['gold']): 
                            outputs[pre][query] = item['gold']
                else:
                    query = ""
                    for subkey in item.keys():
                        if 'label: ' in subkey:
                            query += item[subkey]['testing input']
                    if check_prediction(check_prediction_func_name, query, item['prediction'], item['gold']):
                        outputs[pre][query] = item['gold']

        def self_strip(s):
            if s[0] == "'" or s[0] == '"':
                s = s[1:]
            if s[-1] == "'" or s[-1] == '"':
                s = s[:-1]
            return s

        for key, value in outputs.items():
            out = ""
            for query, answer in value.items():
                out += self_strip(repr(query))+self_strip(repr(answer))+'\n'
            with open(f"{BAD_CASE_INSTRUCTIONS_DIRNAME}/{key}.instructions", 'w', encoding='utf-8') as output_file:
                output_file.write(out)
    #             break
    # break

