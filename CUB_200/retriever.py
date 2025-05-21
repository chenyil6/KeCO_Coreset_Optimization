import torch
import random
from utils import get_imagenet_prompt
import os
import json
import numpy as np
import time
from collections import defaultdict, deque
import logging
import torch.nn.functional as F
import math
from utils import prepare_prompt_mc

logger = logging.getLogger(__name__)

class DynamicReteiever:
    def __init__(self, args):
        self.args = args
        self.demonstrations = []
        self.label2sample = dict()
        self.pool_label2sample = dict()
        self.most_similar = {}
        self.pool = []
        self.label_to_prototype = {}
        self.test_idx2_ice_idx={}

    def get_final_query(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        ice_text = ""
        ice_img=[]
        if demonstrations is not None:
            for dm in demonstrations:
                ice_img.append(dm.image)
                ice_text += get_imagenet_prompt(dm.label)

        ice_img.append(sample.image)
        ice_text += get_imagenet_prompt()
        return ice_img,ice_text,demonstrations
    
    def get_final_query_qwen(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        ice_text = []
        ice_img=[]
        if demonstrations is not None:
            for dm in demonstrations:
                ice_img.append(dm.image)
                prompt, gold_label, label2option = prepare_prompt_mc(dm.options,dm.label)

                ice_text.append(prompt)

        ice_img.append(sample.image)
        prompt_t, gold_label_t, label2option_t = prepare_prompt_mc(sample.options)
        ice_text.append(prompt_t)
        return ice_img,ice_text,demonstrations,gold_label_t,label2option_t

    def get_final_query_idev2(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        ice_text = ""
        ice_img=[]
        if demonstrations is not None:
            for dm in demonstrations:
                ice_img.append(dm.image)
                prompt, gold_label, label2option = prepare_prompt_mc(dm.options,dm.label)
                ice_text+="<image>"
                ice_text += prompt

        if self.args.dnum == 0:
            ice_img = []
        ice_img.append(sample.image)
        prompt_t, gold_label_t, label2option_t = prepare_prompt_mc(sample.options)
        ice_text+="<image>"
        ice_text += prompt_t
        return ice_img,ice_text,demonstrations,gold_label_t,label2option_t
    
    def get_demonstrations_from_bank(self, sample):
        if self.args.dnum == 0:
            return []
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "cosine":
            indices = self.get_topk_cosine(sample)
        elif self.args.select_strategy == "diversity":
            indices = self.get_topk_diverse(sample)
        else:
            print("select_strategy is not effective.")
            return
        return [self.demonstrations[i] for i in indices]

    def get_random(self, sample):
        random.seed(self.args.seed+sample.idx)
        indices = random.sample(range(len(self.demonstrations)), self.args.dnum)
        return indices

    def get_topk_cosine(self, sample):
    
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
       
        sample_embed = sample.feature.to(device).unsqueeze(0)  

  
        demonstration_embeds = torch.stack([sample.feature for sample in self.demonstrations]).to(device)  

        scores = torch.cosine_similarity(demonstration_embeds, sample_embed, dim=1)  # (N,)

      
        values, indices = torch.topk(scores, self.args.dnum, largest=True)  
        indices = indices.cpu().tolist()  
        #print("indices:",indices) 
        return indices

    def get_topk_diverse(self, sample):

        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
   
        sample_embed = sample.feature.to(device).unsqueeze(0)  


        demonstration_embeds = torch.stack([sample.feature for sample in self.demonstrations]).to(device)

        scores = torch.cosine_similarity(demonstration_embeds, sample_embed, dim=1) 

   
        values, indices = torch.topk(scores, self.args.dnum, largest=False)
        indices = indices.cpu().tolist()  
        #print("indices:",indices) 
        return indices
    
    def update_online(self,query_sample):
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        label = query_sample.label
        
        sample_list = self.label2sample[label]
        if self.args.target_select == "random":
            # 随机选择当前类别的一个样本作为目标样本
            target_sample = random.choice(self.label2sample[query_sample.label])

        elif self.args.target_select == "most_similarity":
            query_embedding = query_sample.feature.to(device)  # (256, 1024)

            # 获取当前类别所有样本的特征
            embeddings = torch.stack([s.feature for s in sample_list])  # (N, 256, 1024)
            embeddings = embeddings.to(device)

            # 计算与 query_sample 的余弦相似度
            similarities = F.cosine_similarity(embeddings, query_embedding.unsqueeze(0), dim=-1)  # (N, 256)
            mean_similarities = similarities.mean(dim=-1)  # 每个样本与 query_sample 的平均相似度 (N,)
            
            # 找到最相似的样本
            most_similar_index = torch.argmax(mean_similarities).cpu().item()
            target_sample = sample_list[most_similar_index]
        elif self.args.target_select == "least_similarity":
            query_embedding = query_sample.feature.to(device)  # torch.Size([512])
            # 获取当前类别所有样本的特征
            embeddings = torch.stack([s.feature for s in sample_list])  # (N, 512)
            embeddings = embeddings.to(device)

            # 计算与 query_sample 的余弦相似度
            similarities = F.cosine_similarity(embeddings, query_embedding.unsqueeze(0), dim=-1)  # (N,)
            
            least_similar_index = torch.argmin(similarities).cpu().item()  # 使用 similarities 而不是 mean_similarities
            target_sample = sample_list[least_similar_index]
        
        self.update_based_on_fixed(target_sample,query_sample,self.args.alpha)
            
        
    def update_based_on_fixed(self,target_sample,query_sample,alpha=0.2):   
 
        target_sample.feature = (1-alpha) * target_sample.feature + alpha * query_sample.feature
        

    def update_offline(self,):
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        batch_size = self.args.num
        num_batches = len(self.pool) // batch_size
        batches = [self.pool[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        epoch= self.args.epoch
        # 遍历每个 batch
        for e in range(epoch):
            for batch in batches:
                m2p = defaultdict(lambda: defaultdict(list))
                for b_sample in batch:
                    label = b_sample.label
                    m_list = self.label2sample[label]
                    m_list_features = torch.stack([m.feature for m in m_list]).to(device)
                    b_feature = b_sample.feature.to(device)
                    # 找到 m_list 中 和b_sample最不相似的样本
                    if self.args.target_select == "least_similarity":
                        cos_similarities = torch.cosine_similarity(m_list_features, b_feature.unsqueeze(0), dim=-1)

                        # 找到最不相似的样本索引
                        target_idx = torch.argmin(cos_similarities).item()
                    elif self.args.target_select == "random":
                        # 从 m_list 中随机选一个
                        target_idx = random.choice(range(len(m_list)))
                    elif self.args.target_select == "most_similarity":
                        cos_similarities = torch.cosine_similarity(m_list_features, b_feature.unsqueeze(0), dim=-1)

                        # 找到最相似的样本索引
                        target_idx = torch.argmax(cos_similarities).item()

                    if label not in m2p:
                        m2p[label][target_idx] = [b_sample.feature]
                    else:
                        m2p[label][target_idx].append(b_sample.feature)
                        

                
                # 遍历完 一整个batch 后， 进行更新
                for label, dic in m2p.items():
                    #print("label:",label)
                    #print(dic)
                    for idx,p_list in dic.items():
                        gradient = 0
                        target_f  = self.label2sample[label][idx].feature
                        for p in p_list:
                            gradient += (target_f - p)
                        
                        # 累加完之后，求平均，再用公式更新
                        gradient = gradient/len(p_list)
                        self.label2sample[label][idx].feature = self.label2sample[label][idx].feature-self.args.alpha*gradient

        assert len(self.demonstrations) == self.args.M
        