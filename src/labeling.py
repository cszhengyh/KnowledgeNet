# python /share/project/zhengyuhui/domain_tree_3.1.1/labeling.py > /share/project/zhengyuhui/domain_tree_3.1.1/log/labeling.log
# SAMPLE_SFT_DATA_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/sample_sft_data.jsonl"

# python /share/project/zhengyuhui/domain_tree_3.1.1/labeling.py > /share/project/zhengyuhui/domain_tree_3.1.1/log/benchmark_labeling.log
SAMPLE_SFT_DATA_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/benchmark_instructions/bad_case_instruction/GPQA_diamond.instructions"

import os
import re
import openai
from tqdm import tqdm
import jsonlines
from collections import defaultdict
from prompts.label_prompt import label_prompt
from chat_with_gpt import chat_with_gpt
from domain_tree.layers.layer_1 import *
from domain_tree.layers.layer_2 import *
from domain_tree.layers.layer_3 import *
import matplotlib.pyplot as plt

messages = []
openai.api_key = "EMPTY"
openai.base_url = "http://127.0.0.1:7999/v1/"
model = "Llama3_1-70B-6M-math-0729-megatron"

label_to_instructions = defaultdict(lambda: [])
sft_data_instructions = []
instructions_file = ['lukaemon_mmlu']
expand_instructions = []

# with jsonlines.open(SAMPLE_SFT_DATA_PATH, 'r') as sft_data:
#     for line in sft_data:
#         sft_data_conversations = line['conversations']
#         temp = ""
#         for sft_data_conversation in sft_data_conversations:
#             temp += sft_data_conversation['value']
#         sft_data_instructions.append(temp)
#         # print(sft_data_conversation['value'])

# BAD_INSTRUCTIONS_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/benchmark_instructions/bad_case_instruction/processed_data"
# # for name in instructions_file:
# for f in os.listdir(BAD_INSTRUCTIONS_PATH):
#     # if f[:f.find('.')][:len(name)] == name:
#     with open(os.path.join(BAD_INSTRUCTIONS_PATH, f), 'r', encoding='utf-8') as sft_data:
#         for line in sft_data:
#             sft_data_instructions.append(line)

BAD_INSTRUCTIONS_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/benchmark_instructions/bad_case_instruction"
for name in instructions_file:
    for f in os.listdir(BAD_INSTRUCTIONS_PATH):
        if f[:f.find('.')][:len(name)] == name:
            with open(os.path.join(BAD_INSTRUCTIONS_PATH, f), 'r', encoding='utf-8') as sft_data:
                for line in sft_data:
                    sft_data_instructions.append(line)

# with open(SAMPLE_SFT_DATA_PATH, 'r', encoding='utf-8') as sft_data:
#     for line in sft_data:
#         sft_data_instructions.append(line)

# test_instruction = """Question: Which event most likely allowed for the explosion of mammal diversity that occurred during the Cretaceous period?\nA. formation of the supercontinent Pangea\nB. intense volcanic activity\nC. cooler temperatures\nD. rising sea levels\nAnswer:C"""
# sft_data_instructions.append(test_instruction)

def labeling(sft_data_instructions):
    def get_labels(label_set_names):
        labels = ""
        for label_set_name in label_set_names:
            try:
                label_set_value_name = re.sub(r'[- ()]', '_', label_set_name)+"_labels"
                for label in globals()[label_set_value_name]:
                    labels += label+", "
            except Exception as e:
                # print(e)
                continue
        return labels
    
    label_path = defaultdict(lambda: [])

    LABEL_LEVEL = 4
    MAX_RETRY_CNT = 5
    MAX_INPUT_LEN = 2048
    for sft_data_instruction in tqdm(sft_data_instructions):
        layer_targets = []

        print("\n=========================================\n")
        label_set_names = ['base']
        for level in range(0, LABEL_LEVEL):
            retry_cnt = 0
            parent_label_set_names = label_set_names
            labels = get_labels(label_set_names)[:-2]
            if labels == "":
                break

            query = label_prompt + "\n\n" + f"Now, please respond following the above requirements and examples, the multiple knowledge categories are \"{labels}\", the text is \"{sft_data_instruction}."
            messages = [{"role":"user","content": " ".join(query.split(" ")[:MAX_INPUT_LEN])+'"'}]
            print("\n+++++++++++query--------------\n")
            print(f"the text is \"{sft_data_instruction}\", the multiple knowledge categories are \"{labels}\".")        

            none_flag = 0
            while True:
                # print(retry_cnt)
                if retry_cnt > MAX_RETRY_CNT:
                    expand_instructions.append(sft_data_instruction)
                    label_set_names = parent_label_set_names
                    none_flag = 1
                    print(f"Beyond MAX_RETRY_CNT in level {level}!") # 带这个的指令都是可以用来优化标签树的，就是expand指令
                    break
                try:
                    # gpt4o
                    # targets = chat_with_gpt(messages)

                    # llama3 70B
                    response = openai.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.7,
                            top_p=0.95,
                            stop="ation:"
                            )        
                    targets = response.choices[0].message.content
                    
                    if re.findall(r"\[labels:\s*(.*?),\s*Explan", targets):
                        targets = re.findall(r"\[labels:\s*(.*?),\s*Explan", targets)[0]
                        if targets == 'none':
                            print("None Retry!")
                            retry_cnt += 1
                            continue
                        label_set_names = list(set(targets.lower().split(", ")))

                        flag = 0
                        for target in label_set_names:
                            if target not in [label.lower() for label in labels.split(", ")]:
                                print(f"Target {target} Error Retry!")
                                flag = 1
                                break
                        if flag:
                            continue

                        print("\n+++++++++++target--------------\n")
                        print(label_set_names)
                        layer_targets.append(label_set_names)
                        # print(response.choices[0].message.content)
                        # print([label.lower() for label in labels.split(", ")])
                        break
                    else:
                        continue
                except Exception as e:
                    print(e)
                    continue

            if none_flag:
                break

        print("\n============all targets===========\n")
        print(label_set_names)
        for label in label_set_names:
            label_to_instructions[label].append(sft_data_instruction)
        
        def show_label_path(init_label, label, idx, path):
            if idx < 0:
                if path not in label_path[init_label]:
                    label_path[init_label].append(path)
                return 

            for target in layer_targets[idx]:
                # print("----------------")
                # print(label, target, get_labels([target])[:-2].split(", "))
                if label in get_labels([target])[:-2].lower().split(", "):
                    path.append(target)
                    show_label_path(init_label, target, idx-1, path)
                    path = path[:-1]

        print("\n=============labels path==================\n")
        for label in label_set_names:
            if len(label_path[label]) != 0:
                print(f"{label} is existing! {label}'s label_path is {label_path[label]}.")
                continue

            show_label_path(label, label, len(layer_targets)-2, [])
            print(f"{label}'s label_path is {label_path[label]}")

    sort_label_to_instructions = dict(sorted(label_to_instructions.items(), key=lambda item:len(item[1]), reverse=True))
    Y = []
    for key, value in sort_label_to_instructions.items():
        Y.append(len(value))
        print(f"\n===============================\nlabel: {key}, label_path_len: {[len(item) for item in label_path[key]]}, label_path: {label_path[key]}, count: {len(value)}\n[")
        for instruction in value:
            print(instruction)
    
    print("\n===========================expand instructions===========================\n")
    print(f"Expand instructions number is {len(expand_instructions)}.")
    for instruction in expand_instructions:
        print(instruction)

labeling(sft_data_instructions)

# plt.bar(list(sort_label_to_instructions.keys()), Y)
# plt.xlabel('Domains')
# plt.xticks(rotation=90)
# plt.ylabel('Number')
# plt.title('Domain Distribution')
# plt.show()

'''
计算机科学，数学，环境科学，信息科学，系统科学，生物学，管理学，艺术，语言学，心理学，食品科学，能源科学，化学
'''