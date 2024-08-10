import re
import sys
sys.path.append("/share/project/zhengyuhui/domain_tree_3.1.1")

from chat_with_gpt import chat_with_gpt
from prompts.generate_subdiscipline_prompt import generate_subdiscipline_prompt
from domain_tree.layers.layer_1 import *
from domain_tree.layers.layer_2 import *
from domain_tree.layers.layer_3 import *

def get_labels(label_set_names):
    labels = ""
    for label_set_name in label_set_names:
        # print(label_set_name)
        # import pdb
        # pdb.set_trace()
        try:
            label_set_value_name = re.sub(r'[- ()]', '_', label_set_name)+"_labels"
            # print(label_set_value_name)
            for label in globals()[label_set_value_name]:
                labels += label+", "
        except Exception as e:
            # print(e)
            continue
    return labels

root = ['base']
expand_layer = 2
cur_layer_labels = root

for layer_num in range(0, expand_layer):
    cur_layer_labels = get_labels(cur_layer_labels)[:-2].lower().split(', ')
    # break

expand_labels = []
for label in cur_layer_labels:
    if get_labels([label]) == "":
        expand_labels.append(label)
print(expand_labels)

expand_labels = ['algebraic geometry']
print(f"\n=============expand layer {expand_layer}==================\n")
temp_generate_subdiscipline_prompt = generate_subdiscipline_prompt
for label in expand_labels:
    # print(generate_subdiscipline_prompt.replace('<LABEL>', label))
    while True:
        response = chat_with_gpt(generate_subdiscipline_prompt.replace('<LABEL>', label))
        # print(response)
        if re.findall(r'\[<DIS#>:\s*(.*?)\]', response):
            subdisciplines = re.findall(r'\[<DIS#>:\s*(.*?)\]', response)[0]
            # print(subdisciplines)
            break
        else:
            continue
    format_subdisciplines = ""
    for item in subdisciplines.split(', '):
        format_subdisciplines += f'"{item}", '
    print(f"{re.sub(r'[- ()]', '_', label)}_labels = [{format_subdisciplines[:-2]}]")
    # break
            
