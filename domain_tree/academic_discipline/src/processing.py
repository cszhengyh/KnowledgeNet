# python /share/project/zhengyuhui/domain_tree_3.1.1/hierarchical_outline/processing.py > /share/project/zhengyuhui/domain_tree_3.1.1/log/process_log
RAW_DATA_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/domain_tree/academic_discipline/raw_academic_discipline_zh.txt"
PROCESS_DATA_OUT_PATH = "/share/project/zhengyuhui/domain_tree_3.1.1/log/process.log"

'''
1: 62, 2: 677, 3: 2360, tot: 3270
'''

import re

with open(RAW_DATA_PATH, 'r', encoding='utf-8') as raw_data_file:
    with open(PROCESS_DATA_OUT_PATH, 'a', encoding='utf-8') as out:
        for line in raw_data_file:
            if "其他" in line or "其它" in line or line.strip() == '' or line.strip() == "表1 （续）" or line.strip() == "代 码 学 科 名 称 说 明" or re.match(r'^\d+$', line.strip()):
                continue
            else:
                out.write(line)
    