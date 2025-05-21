from cub200_dataset import CUB200Dataset
from tqdm import tqdm
import pickle
import numpy as np
from transformers import AutoTokenizer, CLIPModel,AutoProcessor
import torch
from classification_utils import CUB_CLASSNAMES
import json
import random
from transformers import AutoProcessor, AutoModel
from utils import get_topk_classifications

import sys
sys.path.append('/path/to/KeCo')
from open_flamingo_v2.open_flamingo.src.factory import create_model_and_transforms

device = "cuda:1"

model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="mpt/mpt-7b",
            tokenizer_path="mpt/mpt-7b",
            cross_attn_every_n_layers=4,
            precision="fp16",
            inference=True,
            device=device,
            checkpoint_path="/path/to/checkpoint.pt"
        )

train_dataset = CUB200Dataset(
            root="/path/to/CUB_200_2011",index_file="./class_to_indices_train.pkl"
        )
test_dataset = CUB200Dataset(
    root="/path/to/CUB_200_2011",
    train=False
)

support_set = []
sample_pool = []

def get_imagenet_prompt(label=None) -> str:
    return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

def prepare_text(ice_text):
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(ice_text, return_tensors="pt")
    lang_x = {k: v.to(device) for k, v in lang_x.items()}
    return lang_x

def prepare_image(img_list):
    vision_x = [image_processor(img).unsqueeze(0) for img in img_list]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    vision_x = vision_x.to(device).half()
    return vision_x

def get_p1(ice_sample,query_sample):
    ice_img = [query_sample["image"]]
    ice_text =get_imagenet_prompt(ice_sample["class_name"])
    ice_text += get_imagenet_prompt()
    
    vision_x = prepare_image(ice_img)
    lang_x = prepare_text(ice_text)
    with torch.no_grad():
        outputs = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            min_new_tokens=1,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True
        )

    classnames_tokens = tokenizer(
            CUB_CLASSNAMES
        )["input_ids"]

    overall_log_probs = get_topk_classifications(outputs,classnames_tokens,topk=1)
    # compute accuracy
    y_i = query_sample["class_name"]

    # Find the index of the ground truth label
    gt_label_index = CUB_CLASSNAMES.index(y_i)

    # Get the confidence score of the ground truth label
    gt_label_confidence = overall_log_probs[gt_label_index].item()

    return gt_label_confidence

def get_p2(ice_sample,query_sample):
    ice_img = [ice_sample["image"],query_sample["image"]]
    ice_text =get_imagenet_prompt(ice_sample["class_name"])
    ice_text += get_imagenet_prompt()
    
    vision_x = prepare_image(ice_img)
    lang_x = prepare_text(ice_text)
    with torch.no_grad():
        outputs = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            min_new_tokens=1,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True
        )

    classnames_tokens = tokenizer(
            CUB_CLASSNAMES
        )["input_ids"]

    overall_log_probs = get_topk_classifications(outputs,classnames_tokens,topk=1)
    # compute accuracy
    y_i = query_sample["class_name"]

    # Find the index of the ground truth label
    gt_label_index = CUB_CLASSNAMES.index(y_i)

    # Get the confidence score of the ground truth label
    gt_label_confidence = overall_log_probs[gt_label_index].item()

    return gt_label_confidence

def compute_contribute(e_i,e_prime): # 计算 e_i当ice后 对 e_prime的贡献量
    p1 = get_p1(e_i,e_prime)
    p2 = get_p2(e_i,e_prime)
    return p2-p1

def infoscore_filter(data_list, m, rho, l):
    D_prime = data_list
    S = random.sample(data_list, l)

    while len(D_prime) > m:
        scores = []
        for e_i in D_prime:
            score = 0
            # 计算 e_i 的信息量，即 e_i 对 集合 S的 信息贡献
            for e_prime in S:
                contribute = compute_contribute(e_i,e_prime)
                score += contribute
            scores.append(score)

        if len(D_prime) / rho < m:
            D_prime = [D_prime[i] for i in np.argsort(scores)[-m:]]
            break
        else:
            D_prime = [D_prime[i] for i in np.argsort(scores)[-int(len(D_prime) / rho):]]

        S_prime = random.sample(data_list, l * (rho - 1))
        S.extend(S_prime)

    return D_prime

sample_method  = "infoscore"

support_set = []
sample_pool =[]


selected_samples = []
remaining_samples= []

for class_id in tqdm(range(200),desc="get samples for each class"):
    data_list = train_dataset.get_data_list_by_class(class_id=class_id)[0:25]

    selected_samples = infoscore_filter(data_list, m=5, rho=3, l=10)
    remaining_samples = [sample for sample in data_list if sample not in selected_samples]     

    # 预处理并添加到 support_set
    for s in selected_samples: 
        support_set.append(s["id"])

    for s in remaining_samples:
        sample_pool.append(s["id"])


print(f"Selected samples: {len(support_set)}")
print(f"Remaining samples: {len(sample_pool)}")


support_set_path = f"./sample/{sample_method}_support_set.json"
sample_pool_path = f"./sample/{sample_method}_sample_pool.json"

with open(support_set_path, 'w') as f:
    json.dump(support_set, f)

with open(sample_pool_path, 'w') as f:
    json.dump(sample_pool, f)

print(f"Support set saved to {support_set_path}")
print(f"Sample pool saved to {sample_pool_path}")