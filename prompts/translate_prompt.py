# 你是一个学术学科方面的资深助理，你将被提供一些中文学术学科，你的任务是将它们翻译为英文，并按照["A", "B", ......]的格式响应。例如，1404510 量子电子学，1404520 电子离子与真空物理，1404530 带电粒子光学，你的回答应该是["Quantum Electronics", "Electron Ionization and Vacuum Physics", "Charged Particle Optics"]。
# [item.strip('"')+"_labels" for item in s.lower().replace(' ', '_').split(",_")]

import re
import sys
import openai
sys.path.append("/share/project/zhengyuhui/domain_tree_3.1.1")
from chat_with_gpt import chat_with_gpt

translate_prompt = r"""You are a senior assistant in academic disciplines and in the field of translation. You will be provided with a Chinese expression of an academic discipline. Your task is to explain the academic discipline step by step and translate it into the official english academic discipline terms of Wikipedia, Baidu Baike, etc. Deliver your response in this format: [translation: ......]. Here are some examples.
Example 1. if the Chinese expression for the academic discipline is "渗透力学," your response should be [translation: Poromechanics], not [translation: Permeability Mechanics].
Example 2. if the Chinese expression for the academic discipline is "结合代数," your response should be [translation: Associative Algebra].
"""

if __name__ == "__main__":
    print(chat_with_gpt(translate_prompt+'The Chinese expression for the academic discipline is 口腔矫形学.'))