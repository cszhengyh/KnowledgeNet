import re
import openai
from collections import defaultdict
from chat_with_gpt import chat_with_gpt
from prompts.translate_prompt import translate_prompt

def translate(zh_discipline):
    # return zh_discipline
    query = translate_prompt + f"Now, the Chinese expression of the academic discipline is \"{zh_discipline}\", please respond in accordance with the above requirements."

    # gpt4o
    while True:
        response = chat_with_gpt(query)
        if re.findall(r'\[translation: \s*(.*?)\]', response):
            translation = re.findall(r'\[translation: \s*(.*?)\]', response)[0]
            break
        else:
            continue
    
    translation = re.sub(r', and ', ' and ', translation)
    translation = re.sub(r', ', ' and ', translation)
    print(zh_discipline, translation)
    return translation

discipline_len_to_layer_id = {'3': 'layer_2', '5': 'layer_3', '7': 'layer_4', '9': 'layer_5'}
layer_id_to_discipline_ids = defaultdict(lambda: [])
discipline_id_to_discipline_name = defaultdict(lambda: -1)
discipline_ids, son_disciplines = [], defaultdict(lambda: [])

with open(NEW_ZH_DISCIPLINE_PATH, 'r', encoding='utf-8') as disciplines:
    for discipline in disciplines:
        print(discipline)
        discipline = discipline.strip().split(' ')
        print(discipline)
        assert len(discipline) == 2, f"{discipline[0]} length error!"

        discipline_id, discipline_name = discipline[0], discipline[1]
        discipline_ids.append(discipline_id)
        layer_id = discipline_len_to_layer_id[str(len(discipline_id))]

        assert discipline_id not in layer_id_to_discipline_ids[layer_id], f"duplicate discipline id: {discipline_id}!"
        layer_id_to_discipline_ids[layer_id].append(discipline_id)
        discipline_id_to_discipline_name[discipline_id] = translate(discipline_name)

for discipline_id in discipline_ids:
    for son_discipline_id in discipline_ids:
        if len(son_discipline_id) == len(discipline_id)+2:
            if son_discipline_id[:len(discipline_id)] == discipline_id:
                son_disciplines[discipline_id].append(son_discipline_id)

out = "base_labels = ["
for discipline_id in layer_id_to_discipline_ids['layer_2']:
    discipline_name = '"' + discipline_id_to_discipline_name[discipline_id] + '", '
    out += discipline_name
with open(f"{LAYERS_PATH}/layer_1.py", 'a', encoding='utf-8') as layer_1_out:
    layer_1_out.write(out[:-2]+"]")

for layer_id in discipline_len_to_layer_id.values():
    out = ""
    for discipline_id in layer_id_to_discipline_ids[layer_id]:
        if len(son_disciplines[discipline_id]) == 0:
            continue
        discipline_name = discipline_id_to_discipline_name[discipline_id]
        out += f"{re.sub(r'[ ()-]', '_', discipline_name.lower())}_labels = ["
        for son_discipline_id in son_disciplines[discipline_id]:
            out += '"' + discipline_id_to_discipline_name[son_discipline_id] + '", '
        out = out[:-2]+"]\n"
    with open(f"{LAYERS_PATH}/{layer_id}.py", 'a', encoding='utf-8') as layer_out:
        layer_out.write(out)

