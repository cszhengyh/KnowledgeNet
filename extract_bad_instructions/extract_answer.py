def extract_lukaemon_mmlu_answer(prediction):
    return prediction

# def extract_agieval_answer():


def extract_ARC_c_answer(prediction):
    return prediction[0]

def extract_GPQA_answer(prediction):
    import re
    return re.findall(r'\((.*?)\)', prediction)[0]

def extract_ceval_answer(prediction):
    return prediction

# def extract_cmmlu_answer():


# def extract_Gaokao_answer():
    