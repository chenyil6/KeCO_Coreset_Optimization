import torch
from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_100_CLASS_ID_TO_LABEL
import math
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_imagenet_prompt(label=None) -> str:
    return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

def get_topk_classifications(outputs, classnames_tokens, topk=2, temperature=2.0):
    overall_log_probs = torch.zeros(len(classnames_tokens))
    for idx,ct in enumerate(classnames_tokens):
        classname_tokens_num = len(ct)
        log_prob = 0
        valid = True  # 用于标记生成的token是否足够覆盖类别名称
        for i in range(classname_tokens_num):
            try:
                #  Compute log probabilities and probabilities
                log_scores = torch.nn.functional.log_softmax(outputs.scores[i]/temperature, dim=-1).squeeze().tolist()
                # 检查当前的 log_scores 是否有效
                if log_scores[ct[i]] == -float('inf'):
                    valid = False
                    break  # 如果有无效的 log score，跳出循环
                # Sum the log probabilities instead of multiplying probabilities
                log_prob += log_scores[ct[i]]
            except IndexError as e:
                #print(f"IndexError encountered at position {i} with ct[i]={ct[i]} and token={ct[i]}: {str(e)}")
                log_prob = -float('inf')
                valid = False
                break  # Exit the loop if there's an IndexError
        
            if valid:
                log_prob /= classname_tokens_num
                overall_log_probs[idx] = torch.exp(torch.tensor(log_prob))   # 将log概率转换为概率

    # 归一化类别概率
    overall_log_probs = overall_log_probs / overall_log_probs.sum()

    predicted_classnames, predicted_probs = get_predicted_classname(
        overall_log_probs,
        k=topk,
        class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL,
    )
    return predicted_classnames, predicted_probs, overall_log_probs

import torch

def get_topk_classifications_batch(outputs, classnames_tokens, topk, class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL):
    """
    Computes the top-k classifications for a batch of samples.

    Args:
        outputs: The outputs from model.generate, which includes scores.
        classnames_tokens: List of lists, tokenized class names.
        topk: int, number of top predictions to return.
        temperature: float, temperature parameter for scaling logits.
        class_id_to_name: A dictionary mapping class indices to class names.

    Returns:
        predicted_classnames: List of lists containing predicted class names for each sample.
        predicted_probs: List of lists containing predicted probabilities for each sample.
        overall_probs: Tensor of shape (batch_size, num_classes) containing the normalized probabilities.
    """
    batch_size = outputs.sequences.shape[0]
    num_classes = len(classnames_tokens)
    overall_log_probs = torch.zeros(batch_size, num_classes)

    for sample_idx in range(batch_size):
        for class_idx, ct in enumerate(classnames_tokens):
            classname_tokens_num = len(ct)
            log_prob = 0
            valid = True
            
            for i in range(classname_tokens_num):
                try:
                    log_scores = torch.nn.functional.log_softmax(outputs.scores[i][sample_idx], dim=-1)
                    log_prob += log_scores[ct[i]]
                except IndexError as e:
                    #print(f"IndexError at position {i} with ct[i]={ct[i]} and token={ct[i]}: {str(e)}")
                    log_prob = -float('inf')
                    valid = False
                    break
            
            if valid:
                log_prob /= classname_tokens_num
                overall_log_probs[sample_idx, class_idx] = log_prob

    # Apply softmax across classes to get probabilities
    overall_probs = torch.exp(overall_log_probs)  # Convert log-probs to probs
    overall_probs = overall_probs / overall_probs.sum(dim=-1, keepdim=True)

    # Get top-k predictions for each sample
    topk_probs, topk_indices = torch.topk(overall_probs, topk, dim=-1)
    predicted_classnames = []
    predicted_probs = []

    for sample_idx in range(batch_size):
        sample_classnames = [class_id_to_name[idx.item()] if class_id_to_name else idx.item() for idx in topk_indices[sample_idx]]
        sample_probs = topk_probs[sample_idx].tolist()
        predicted_classnames.append(sample_classnames)
        predicted_probs.append(sample_probs)

    return predicted_classnames, predicted_probs, overall_probs


def get_predicted_classname(logprobs, k, class_id_to_name):
    """
        Args:
            - logprobs: list containing logprobs for each classname
            - k: number for top-k
            - class_id_to_name: dict mapping class index to classname

        Returns:
            - top-k predicted classnames list type str
            - top-k logprobs list type float
        """
    values, indices = torch.topk(logprobs, k=k, dim=0)  # shape (k,)

    predicted_classnames = [class_id_to_name[ix.item()] for ix in indices]
    predicted_logprobs = values.tolist()

    return predicted_classnames, predicted_logprobs

def prepare_prompt_mc( options, answer=None,
                   prompt_query="Which of these choices is shown in the image?",
                   prompt_options="\nChoices:\n",  #Choices:
                   prompt_end="\nAnswer with the letter from the given choices directly.",
                   choice_enumeration="ABCD"):  #\nAnswer with the letter from the given choices directly.

    # print("options:",options)
    # print("answer:",answer)
    shuffle_idx = np.random.permutation(len(options)).tolist() 
    gold_idx = shuffle_idx.index(0) 
    gold_label = choice_enumeration[gold_idx] 
    #print("gold_label:",gold_label)
    formatted_options = "\n".join([f"{choice_enumeration[i]}. {options[j]}" for i, j in enumerate(shuffle_idx)])
    #print("formatted_options:",formatted_options)
    label2option = {choice_enumeration[i]: options[j] for i, j in enumerate(shuffle_idx)}
    formatted_prompt = f"{prompt_query}{prompt_options}{formatted_options}{prompt_end}{gold_label if answer is not None else ''}{'.' if answer is not None else ''}"
    #print("formatted_prompt:",formatted_prompt)
    return formatted_prompt, gold_label, label2option

import re

def extract_last_answer(text):

    answers = re.findall(r'Answer with the letter from the given choices directly\.([A-D])\.', text)
    #print(answers)
    # 使用正则表达式分割文本
    answer = answers[-1]
    #print("answer:",answer)
    return answer

def extract_letters_from_output(output_texts):
    """
    从输出列表中提取每个答案对应的字母（A/B/C/D）。
    
    参数:
        output_texts (list): 包含答案的列表，例如 ['C.', 'B.', 'A.']。
    
    返回:
        list: 提取的字母列表，例如 ['C', 'B', 'A']。
    """
    print(f"output_text:{output_texts}")
    if not output_texts:
        raise ValueError("输出列表为空，无法提取字母。")
    
    extracted_letters = []
    
    for text in output_texts:
        # 去除可能的标点符号，提取字母部分
        cleaned_letter = ''.join(filter(str.isalpha, text))
        
        # 检查是否是有效的字母
        if cleaned_letter not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"提取的字母无效：{cleaned_letter}")
        
        extracted_letters.append(cleaned_letter)
    
    return extracted_letters

# 示例
# output_texts = ['C.', 'B.', 'A.']
# result = extract_letters_from_output(output_texts)
# print(result)  # 输出：['C', 'B', 'A']