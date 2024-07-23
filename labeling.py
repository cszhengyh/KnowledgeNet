# python /share/project/zhengyuhui/domain_tree_3.1.1/labeling.py > /share/project/zhengyuhui/domain_tree_3.1.1/log/labeling.log
# SAMPLE_SFT_DATA_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/sample_sft_data.jsonl"

# python /share/project/zhengyuhui/domain_tree_3.1.1/labeling.py > /share/project/zhengyuhui/domain_tree_3.1.1/log/benchmark_labeling.log
SAMPLE_SFT_DATA_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/datasets/datasets_instructions/wiki_science_instructions.out"

import re
import openai
from tqdm import tqdm
import jsonlines
from collections import defaultdict
from prompts.label_prompt import label_prompt
from domain_tree.layers.layer_1 import *
from domain_tree.layers.layer_2 import *
from domain_tree.layers.layer_3 import *
import matplotlib.pyplot as plt

messages = []
openai.api_key = "EMPTY"
openai.base_url = "http://127.0.0.1:7999/v1/"
model = "Llama3-70B-merge-v1-dedup-0625-dedup-filter-ep3-megatron"

label_to_instructions = defaultdict(lambda: [])
sft_data_instructions = []

# with jsonlines.open(SAMPLE_SFT_DATA_PATH, 'r') as sft_data:
#     for line in sft_data:
#         sft_data_conversations = line['conversations']
#         temp = ""
#         for sft_data_conversation in sft_data_conversations:
#             temp += sft_data_conversation['value']
#         sft_data_instructions.append(temp)
#         # print(sft_data_conversation['value'])


with open(SAMPLE_SFT_DATA_PATH, 'r', encoding='utf-8') as sft_data:
    for line in sft_data:
        sft_data_instructions.append(line)

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

    LABEL_LEVEL = 4
    MAX_RETRY_CNT = 5
    MAX_INPUT_LEN = 2048
    for sft_data_instruction in tqdm(sft_data_instructions):
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
                if retry_cnt > MAX_RETRY_CNT:
                    label_set_names = parent_label_set_names
                    none_flag = 1
                    print(f"Beyond MAX_RETRY_CNT in level {level}!")
                    break
                try:
                    response = openai.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.7,
                            top_p=0.95,
                            stop="ation:"
                            )        
                    targets = response.choices[0].message.content
                    if re.findall(r"\[labels: \s*(.*?),\s*Explan", targets):
                        targets = re.findall(r"\[labels: \s*(.*?),\s*Explan", targets)[0]
                        if targets == 'none':
                            print("None Retry!")
                            retry_cnt += 1
                            continue
                        label_set_names = list(set(targets.lower().split(", ")))

                        flag = 0
                        for target in label_set_names:
                            if target not in [label.lower() for label in labels.split(", ")]:
                                print("Target Error Retry!")
                                flag = 1
                                break
                        if flag:
                            continue

                        print("\n+++++++++++target--------------\n")
                        print(label_set_names)
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

    sort_label_to_instructions = dict(sorted(label_to_instructions.items(), key=lambda item:len(item[1]), reverse=True))
    Y = []
    for key, value in sort_label_to_instructions.items():
        Y.append(len(value))
        print(f"\n===============================\nlabel: {key}, count: {len(value)}")
        for instruction in value:
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