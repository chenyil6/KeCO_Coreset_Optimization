import dataclasses
from retriever_mixup import DynamicReteiever
from tqdm import tqdm
import torch
from cub200_dataset import CUB200Dataset
import os
from PIL import Image
from classification_utils import CUB_CLASSNAMES,CUB_200_CLASS_ID_TO_LABEL
from torch.utils.data import Subset
from typing import Optional
from utils import *
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import pickle
import logging
from typing import List
from sklearn.cluster import KMeans
from collections import defaultdict
from qwen_vl_utils import process_vision_info


logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Sample:
    idx: int
    image: Optional[Image.Image]
    label: str
    feature: torch.Tensor
    options: str
    class_id: int
    pseudo_label: Optional[str]   


class Online_ICL:
    """
    Inference code for Online_ICL. You can inference your data with two steps:
    1). Init:             inferencer = Online_ICL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, tokenizer=None, model=None, image_processor=None,device=None,processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = CUB_CLASSNAMES
        self.processor = processor
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1
        self.embedding_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336').to("cuda:0")
        self.embedding_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.no_kv_caching = False
        with open("./train_data_options.json", 'r') as f:
            self.train_options_dict = json.load(f)
        with open("./test_data_options.json", 'r') as f:
            self.test_options_dict = json.load(f)
       
    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def prepare_image(self,img_list):
        vision_x = [self.image_processor(img).unsqueeze(0) for img in img_list]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(self.device).half()
        return vision_x

    def get_text_embedding(self, label):
        inputs = self.embedding_tokenizer(text=label, padding=True,return_tensors="pt")
        with torch.no_grad():
            text_features = self.embedding_model.get_text_features(**inputs)
        return text_features
    
    def prepare_text(self,ice_text):
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(ice_text, return_tensors="pt")
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        return lang_x
    
    
    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch) # 一个批量有多少图片
        batch_images = None
        for iexample, example in enumerate(batch): 
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(self.device).half()
        return batch_images

    def _prepare_text(
            self,
            batch: List[List[str]],
            padding="longest",
            truncation=True,
            max_length=2000,
        ):
            """
            Tokenize the text and stack them.
            Args:
                batch: A list of lists of strings.
            Returns:
                input_ids (tensor)
                    shape (B, T_txt)
                attention_mask (tensor)
                    shape (B, T_txt)
            """
            self.tokenizer.padding_side = "left"
            encodings = self.tokenizer(
                batch,
                padding=padding,
                truncation=truncation,
                return_tensors="pt",
                max_length=max_length,
            )
            input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            return input_ids, attention_mask

    def evaluate_on_idev2(self,query_sample):
        ice_img,ice_text,demonstrations,gold_label_t,label2option_t = self.retriever.get_final_query_idev2(query_sample)
        # Collect prompts and images
        prompt = [ice_text]

        BAD_WORDS_IDS = self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        EOS_WORDS_IDS = [self.tokenizer.eos_token_id]
        
        device_set = "cuda:" + str(self.args.device)
        inputs = self.processor(images=ice_img, text=prompt, padding=True, truncation=True, return_tensors="pt").to(device_set)
       
        # Generate
        generated_ids = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=2)

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        new_prediction = extract_last_answer(generated_texts[0])
        
        self.predictions.append({"answer": new_prediction,
                                "predict_answer":label2option_t[new_prediction],
                                "ground_truth": query_sample.label,
                                "gold_label": gold_label_t,
                                "label2option": label2option_t,
                                "prompt_text": prompt,
                                "prompt_label": [dm.label for dm in demonstrations]
                                })  
        query_sample.pseudo_label = label2option_t[new_prediction]

    def evaluate_batch_on_OFv2(self, batch_samples):
        batch_images = []
        batch_text = []
        batch_demonstrations = []
        for sample in batch_samples: 
            ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
            batch_images.append(ice_img)
            batch_text.append(ice_text)
            batch_demonstrations.append(demonstrations)
        
        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _vision_x = batch_images

        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=_vision_x, 
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                vision_features=None,
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )


        classnames_tokens = self.tokenizer(self.all_class_names)["input_ids"]

        predicted_classnames_batch, predicted_logprobs_batch, overall_log_probs = get_topk_classifications_batch(
            outputs,
            classnames_tokens,
            self.topk,
        )

        # Process predictions for each sample
        for idx, sample in enumerate(batch_samples):
            y_i = sample.label
            gt_label_index = self.all_class_names.index(y_i)
            gt_label_confidence = overall_log_probs[idx, gt_label_index].item()
            demonstrations = batch_demonstrations[idx]
            self.predictions.append(
                {
                    "id": sample.idx,
                    "gt_label": y_i,
                    "pred_label": predicted_classnames_batch[idx][0],
                    "gt_id": sample.class_id,
                    "pred_score": predicted_logprobs_batch[idx][0],
                    "gt_score": gt_label_confidence,
                    "prompt_text": batch_text[idx],
                    "prompt_label": [dm.label for dm in demonstrations]
                }
            )
            sample.pred_score = predicted_logprobs_batch[idx][0]
            sample.pseudo_label = predicted_classnames_batch[idx][0]


    def make_message(self, image_path_list, text_list):
        diction = {}
        diction["role"] = "user"
        diction["content"] =[]
        for i in range(len(image_path_list)):
            diction["content"].append({"type": "image","image":image_path_list[i]})
            diction["content"].append({"type": "text","text":text_list[i]})

        return diction  
    
    def evaluate_on_qwen2(self,query_sample):
        ice_img,ice_text,demonstrations,gold_label_t,label2option_t = self.retriever.get_final_query_qwen(query_sample)
        messages = self.make_message(ice_img,ice_text)
        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results = extract_letters_from_output(output_text)
        self.predictions.append({"idx":query_sample.idx,
                            "answer": results[0],
                            "predict_answer":label2option_t[results[0]],
                            "ground_truth": query_sample.label,
                            "gold_label": gold_label_t,
                            "label2option": label2option_t,
                            "prompt_label": [dm.label for dm in demonstrations]
                            })  
        
        query_sample.pseudo_label = label2option_t[results[0]]
       

    def inference(self,sample):
        sample = self.preprocess_val(sample) 
        self.test_sample_num += 1
        if self.args.model == "open_flamingo_3b" :
            self.evaluate_on_OFv2(sample)
        if self.args.model == "idefics_v2":
            self.evaluate_on_idev2(sample)
        if self.args.model == "qwen2_vl":
            self.evaluate_on_qwen2(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
        
    def preprocess_train(self, sample):
        idx = sample["id"]  
        label = sample["class_name"]
        class_id = sample["class_id"]
        image = sample["image"]
        image_path = sample["image_path"]
        if str(idx) in self.train_options_dict:
            options = self.train_options_dict[str(idx)]
        else:
            options = None
        feature = self.get_embedding(image)
        if self.args.model == "qwen2_vl" :
            image = image_path
        sample = Sample(idx, image, label, feature,options,class_id,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        image = sample["image"]
        image_path = sample["image_path"]
        options = self.test_options_dict[str(idx)]

        feature = self.get_embedding(image)
        if self.args.model == "qwen2_vl" :
            image = image_path
        sample = Sample(idx, image, label, feature,options,class_id,None)
        return sample      
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)

    def compute_distance(self,feature1, feature2):
        return np.linalg.norm(feature1 - feature2)
    
    def run(self):
        results = {"avg":0}
        train_dataset = CUB200Dataset(
            root="/path/to/CUB_200_2011",index_file="./class_to_indices_train.pkl"
        )
        test_dataset = CUB200Dataset(
            root="/path/to//CUB_200_2011",
            train=False
        )

        print(f"self.args.catergory_num:{self.args.catergory_num}")
        print(f"len of self.all_class_names:{len(self.all_class_names)}")

        random.seed(self.args.seed)
    
        validate_rng = random.Random(self.args.seed)  
        sample_pool_rng = random.Random(self.args.seed + 1) 
        

        sample_pool = []

        print("get memory bank and sample pool ...")
            
        for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
            data_list = train_dataset.get_data_list_by_class(class_id=class_id)[0:25]
            if self.args.sample_method == "random":
                selected_samples = data_list[0:self.args.M // len(self.all_class_names)] 
                remaining_samples = data_list[self.args.M // len(self.all_class_names):25]
            elif self.args.sample_method == "k_center_greedy":
                initial_sample = data_list[0]
                s = [initial_sample]
                while len(s) < self.args.M // len(self.all_class_names):
                    max_distance = -1
                    selected_sample = None
                    for sample in data_list:
                        if sample in s:
                            continue
                        
                        min_distance = float('inf')
                        for selected in s:
                            feature1 = self.train_data_feature_siglip[sample["id"]]
                            feature2 = self.train_data_feature_siglip[selected["id"]]
                            distance = self.compute_distance(feature1, feature2)
                            min_distance = min(min_distance, distance)
                        
                        if min_distance > max_distance:
                            max_distance = min_distance
                            selected_sample = sample
                    s.append(selected_sample)
                
                selected_samples = s
                remaining_samples = [sample for sample in data_list if sample not in selected_samples]
            elif self.args.sample_method == "infoscore":
                break
            else:
                raise ValueError(f"Unsupported sample method: {self.args.sample_method}")         

            
            for s in selected_samples:
                processed_s = self.preprocess_train(s)
                self.retriever.demonstrations.append(processed_s)
                self.process_dict(processed_s)

            sample_pool.extend(remaining_samples)

        if self.args.sample_method == "infoscore": 
            with open("./sample/infoscore_support_set.json", 'r') as f:
                idx_list = json.load(f)
            for idx in idx_list:
                support_sample = train_dataset[idx]
                processed_s = self.preprocess_train(support_sample)
                self.retriever.demonstrations.append(processed_s)
                self.process_dict(processed_s)

            with open("./sample/infoscore_sample_pool.json", 'r') as f:
                pool_idx_list = json.load(f)
            sample_pool = [train_dataset[idx] for idx in pool_idx_list]

        print(f"Get the value of every sample in support set")
        sample_pool_rng.shuffle(sample_pool)  

        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)

        total_samples = len(sample_pool)  
        pbar = tqdm(total=total_samples, desc="Using sample pool to update the support set")
        while sample_pool:  
            sample = sample_pool.pop()
            sample = self.preprocess_train(sample)
            self.retriever.update_online(sample)
            del sample
            pbar.update(1)  

        self.predictions = []
        self.test_sample_num = 0
        self.right_sample_num = 0

        assert len(self.retriever.demonstrations) == self.args.M

        print("size of support set:",len(self.retriever.demonstrations))

        for idx in tqdm(shuffled_indices, desc=f"Inference ImageNet..."):
            self.inference(test_dataset[idx])

        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc

        return results,self.predictions
    
class FewShot:
    """
    Inference code for FewShot. You can inference your data with two steps:
    1). Init:             inferencer = FewShot(**kwargs)
    2). inference:        inferencer.run()
    """
    def __init__(self, args, tokenizer=None, model=None, image_processor=None,device=None,processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.processor = processor
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = CUB_CLASSNAMES
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.no_kv_caching = False
        self.topk = 1       
        self.embedding_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336').to("cuda:0")
        self.embedding_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        with open("./train_data_options.json", 'r') as f:
            self.train_options_dict = json.load(f)
        with open("./test_data_options.json", 'r') as f:
            self.test_options_dict = json.load(f)
    
    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def make_message(self, image_path_list, text_list):
        diction = {}
        diction["role"] = "user"
        diction["content"] =[]
        for i in range(len(image_path_list)):
            diction["content"].append({"type": "image","image":image_path_list[i]})
            diction["content"].append({"type": "text","text":text_list[i]})

        return diction  

    def evaluate_on_qwen2(self,query_sample):
        ice_img,ice_text,demonstrations,gold_label_t,label2option_t = self.retriever.get_final_query_qwen(query_sample)
        messages = self.make_message(ice_img,ice_text)
        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"output_text:{output_text}")
        results = extract_letters_from_output(output_text)
        self.predictions.append({"answer": results[0],
                            "predict_answer":label2option_t[results[0]],
                            "ground_truth": query_sample.label,
                            "gold_label": gold_label_t,
                            "label2option": label2option_t,
                            "prompt_label": [dm.label for dm in demonstrations]
                            })  
        
        query_sample.pseudo_label = label2option_t[results[0]]
        #print(f"sample.label:{query_sample.label} and sample.pseudo_label:{query_sample.pseudo_label}")

    def evaluate_on_idev2(self,query_sample):
        ice_img,ice_text,demonstrations,gold_label_t,label2option_t = self.retriever.get_final_query_idev2(query_sample)
        # Collect prompts and images
        prompt = [ice_text]

        BAD_WORDS_IDS = self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        EOS_WORDS_IDS = [self.tokenizer.eos_token_id]
        
        device_set = "cuda:" + str(self.args.device)
        inputs = self.processor(images=ice_img, text=prompt, padding=True, truncation=True, return_tensors="pt").to(device_set)
       
        # Generate
        generated_ids = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=2)

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        new_prediction = extract_last_answer(generated_texts[0])
        
        self.predictions.append({"idx": query_sample.idx,
                                "answer": new_prediction,
                                "predict_answer":label2option_t[new_prediction],
                                "ground_truth": query_sample.label,
                                "gold_label": gold_label_t,
                                "label2option": label2option_t,
                                "prompt_text": prompt,
                                "prompt_label": [dm.label for dm in demonstrations]
                                })  
        query_sample.pseudo_label = label2option_t[new_prediction]


    def inference(self,sample):
        sample = self.preprocess_val(sample) 
        self.test_sample_num += 1
        if self.args.model == "open_flamingo_3b" :
            self.evaluate_batch_on_OFv2(sample)
        if self.args.model == "idefics_v2":
            self.evaluate_on_idev2(sample)
        if self.args.model == "qwen2_vl":
            self.evaluate_on_qwen2(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
        
    def preprocess_train(self, sample):
        idx = sample["id"]  
        label = sample["class_name"]
        class_id = sample["class_id"]
        image = sample["image"]
        image_path = sample["image_path"]
        if str(idx) in self.train_options_dict:
            options = self.train_options_dict[str(idx)]
        else:
            options = None
        feature = self.get_embedding(image)
        if self.args.model == "qwen2_vl" :
            image = image_path
        sample = Sample(idx, image, label, feature,options,class_id,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        image = sample["image"]
        image_path = sample["image_path"]
        options = self.test_options_dict[str(idx)]

        feature = self.get_embedding(image)
        if self.args.model == "qwen2_vl" :
            image = image_path
        sample = Sample(idx, image, label, feature,options,class_id,None)
        return sample      

    def compute_distance(self,feature1, feature2):

        return np.linalg.norm(feature1 - feature2)


    def run(self):
        results = {"avg": 0}
        train_dataset = CUB200Dataset(
            root="/path/to/CUB_200_2011",index_file="./class_to_indices_train.pkl"
        )
        test_dataset = CUB200Dataset(
            root="/path/to/CUB_200_2011",
            train=False
        )
   
        random.seed(self.args.seed)
        
  
        validate_rng = random.Random(self.args.seed)
        print("get supportng set ...")
        support_set = []
        if self.args.bank == "initial":
            for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
                data_list = train_dataset.get_data_list_by_class(class_id=class_id)
                if self.args.sample_method == "random":
                    selected_samples = data_list[0:self.args.M // len(self.all_class_names)]
                   
                elif self.args.sample_method == "k_center_greedy":
                    initial_sample = data_list[0]
                    s = [initial_sample]
                    while len(s) < self.args.M // len(self.all_class_names):
                        max_distance = -1
                        selected_sample = None
                        for sample in data_list:
                            if sample in s:
                                continue
                
                            min_distance = float('inf')
                            for selected in s:
                                feature1 = self.train_data_feature_siglip[sample["id"]]
                                feature2 = self.train_data_feature_siglip[selected["id"]]
                                distance = self.compute_distance(feature1, feature2)
                                min_distance = min(min_distance, distance)
                
                            if min_distance > max_distance:
                                max_distance = min_distance
                                selected_sample = sample
                        s.append(selected_sample)
                    
                    selected_samples = s
                elif self.args.sample_method == "infoscore":
                    break
                else:
                    raise ValueError(f"Unsupported sample method: {self.args.sample_method}")         

          
                for s in selected_samples:
                    processed_s = self.preprocess_train(s)
                    self.retriever.demonstrations.append(processed_s)
             

            if self.args.sample_method == "infoscore": 
                with open("./sample/infoscore_support_set.json", 'r') as f:
                    idx_list = json.load(f)
                self.retriever.demonstrations = [self.preprocess_train(train_dataset[idx]) for idx in idx_list]
        else:
            for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
                all_data_list = train_dataset.get_data_list_by_class(class_id=class_id)[0:25]

                for data in all_data_list:
                    processed_s = self.preprocess_train(data)
                    self.retriever.demonstrations.append(processed_s)
                    
        print("size of coreset:",len(self.retriever.demonstrations))


        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0
        
        for idx in tqdm(shuffled_indices, desc=f"Inference ImageNet..."):
            self.inference(test_dataset[idx])

        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc
        return results,self.predictions

class Offline_ICL:
    def __init__(self, args, tokenizer, model, image_processor,device=None,processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.processor = processor
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = CUB_CLASSNAMES
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1
        self.pool_label2sample = {}
        self.sample_pool = []
        self.no_kv_caching = False
        self.embedding_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336').to("cuda:0")
        self.embedding_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        with open("./train_data_options.json", 'r') as f:
            self.train_options_dict = json.load(f)
        with open("./test_data_options.json", 'r') as f:
            self.test_options_dict = json.load(f)
       
    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def make_message(self, image_path_list, text_list):
        diction = {}
        diction["role"] = "user"
        diction["content"] =[]
        for i in range(len(image_path_list)):
            diction["content"].append({"type": "image","image":image_path_list[i]})
            diction["content"].append({"type": "text","text":text_list[i]})

        return diction  
    
    def prepare_image(self,img_list):
        vision_x = [self.image_processor(img).unsqueeze(0) for img in img_list]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(self.device).half()
        return vision_x

    def get_text_embedding(self, label):
        inputs = self.embedding_tokenizer(text=label, padding=True,return_tensors="pt")
        with torch.no_grad():
            text_features = self.embedding_model.get_text_features(**inputs)
        return text_features
    
    def prepare_text(self,ice_text):
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(ice_text, return_tensors="pt")
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        return lang_x

    def inference(self,sample):
        sample = self.preprocess_val(sample) 
        self.test_sample_num += 1
        if self.args.model == "open_flamingo_3b" :
            self.evaluate_batch_on_OFv2(sample)
        if self.args.model == "idefics_v2":
            self.evaluate_on_idev2(sample)
        if self.args.model == "qwen2_vl":
            self.evaluate_on_qwen2(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
        
    def evaluate_on_qwen2(self,query_sample):
        ice_img,ice_text,demonstrations,gold_label_t,label2option_t = self.retriever.get_final_query_qwen(query_sample)
        messages = self.make_message(ice_img,ice_text)
        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results = extract_letters_from_output(output_text)
        self.predictions.append({"idx":query_sample.idx,
                            "answer": results[0],
                            "predict_answer":label2option_t[results[0]],
                            "ground_truth": query_sample.label,
                            "gold_label": gold_label_t,
                            "label2option": label2option_t,
                            "prompt_label": [dm.label for dm in demonstrations]
                            })  
        
        query_sample.pseudo_label = label2option_t[results[0]]
       

    def evaluate_on_idev2(self,query_sample):
        ice_img,ice_text,demonstrations,gold_label_t,label2option_t = self.retriever.get_final_query_idev2(query_sample)
        # Collect prompts and images
        prompt = [ice_text]

        BAD_WORDS_IDS = self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        EOS_WORDS_IDS = [self.tokenizer.eos_token_id]
        
        device_set = "cuda:" + str(self.args.device)
        inputs = self.processor(images=ice_img, text=prompt, padding=True, truncation=True, return_tensors="pt").to(device_set)
       
        # Generate
        generated_ids = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=2)

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        new_prediction = extract_last_answer(generated_texts[0])
        
        self.predictions.append({"idx": query_sample.idx,
                                "answer": new_prediction,
                                "predict_answer":label2option_t[new_prediction],
                                "ground_truth": query_sample.label,
                                "gold_label": gold_label_t,
                                "label2option": label2option_t,
                                "prompt_text": prompt,
                                "prompt_label": [dm.label for dm in demonstrations]
                                })  
        query_sample.pseudo_label = label2option_t[new_prediction]


    def preprocess_train(self, sample):
        idx = sample["id"]  
        label = sample["class_name"]
        class_id = sample["class_id"]
        image = sample["image"]
        image_path = sample["image_path"]
        if str(idx) in self.train_options_dict:
            options = self.train_options_dict[str(idx)]
        else:
            options = None
        feature = self.get_embedding(image)
        if self.args.model == "qwen2_vl" :
            image = image_path
        sample = Sample(idx, image, label, feature,options,class_id,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        image = sample["image"]
        image_path = sample["image_path"]
        options = self.test_options_dict[str(idx)]

        feature = self.get_embedding(image)
        if self.args.model == "qwen2_vl" :
            image = image_path
        sample = Sample(idx, image, label, feature,options,class_id,None)
        return sample      
    
    def classify_support(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)
    
    def classify_pool(self,sample):
        if sample.label not in self.retriever.pool_label2sample:
            self.retriever.pool_label2sample[sample.label] = [sample]
        else:
            self.retriever.pool_label2sample[sample.label].append(sample)

    def compute_distance(self,feature1, feature2):

        return np.linalg.norm(feature1 - feature2)
    
    def run(self):
        results = {"avg":0}
        train_dataset = CUB200Dataset(
            root="/path/to/CUB_200_2011",index_file="./class_to_indices_train.pkl"
        )
        test_dataset = CUB200Dataset(
            root="/path/to/CUB_200_2011",
            train=False
        )
        print(f"self.args.catergory_num:{self.args.catergory_num}")
        print(f"len of self.all_class_names:{len(self.all_class_names)}")

        random.seed(self.args.seed)
    
        validate_rng = random.Random(self.args.seed)
        sample_pool_rng = random.Random(self.args.seed + 1)
     
        sample_pool = []
        
        for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
            data_list = train_dataset.get_data_list_by_class(class_id=class_id)[0:25]
            if self.args.sample_method == "random":
                selected_samples = data_list[0:self.args.M // len(self.all_class_names)] 
                remaining_samples = data_list[self.args.M // len(self.all_class_names):25]
            elif self.args.sample_method == "k_center_greedy":
                initial_sample = data_list[0]
                s = [initial_sample]
                while len(s) < self.args.M // len(self.all_class_names):
                    max_distance = -1
                    selected_sample = None
                    for sample in data_list:
                        if sample in s:
                            continue
                  
                        min_distance = float('inf')
                        for selected in s:
                            feature1 = self.train_data_feature_siglip[sample["id"]]
                            feature2 = self.train_data_feature_siglip[selected["id"]]
                            distance = self.compute_distance(feature1, feature2)
                            min_distance = min(min_distance, distance)
                 
                        if min_distance > max_distance:
                            max_distance = min_distance
                            selected_sample = sample
                    s.append(selected_sample)
                
                selected_samples = s
                remaining_samples = [sample for sample in data_list if sample not in selected_samples]
            elif self.args.sample_method == "infoscore":
                break
            else:
                raise ValueError(f"Unsupported sample method: {self.args.sample_method}")         


            for s in selected_samples:
                processed_s = self.preprocess_train(s)
                self.retriever.demonstrations.append(processed_s)
                self.classify_support(processed_s)
                #support_set.append(processed_s)
            sample_pool.extend(remaining_samples)

        if self.args.sample_method == "infoscore": 
            with open("./sample/infoscore_support_set.json", 'r') as f:
                idx_list = json.load(f)
            for idx in idx_list:
                support_sample = train_dataset[idx]
                processed_s = self.preprocess_train(support_sample)
                self.retriever.demonstrations.append(processed_s)
                self.classify_support(processed_s)
                
            with open("./sample/infoscore_sample_pool.json", 'r') as f:
                pool_idx_list = json.load(f)
            sample_pool = [train_dataset[idx] for idx in pool_idx_list]

        print(f"Support set size: {len(self.retriever.demonstrations)}, Sample pool size: {len(sample_pool)}")

        sample_pool_rng.shuffle(sample_pool) 
        print("打乱sample pool...")
        
        
        for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess sample_pool..."):
     
            pool_sample = self.preprocess_train(sample_pool[idx])
            self.retriever.pool.append(pool_sample)
            self.classify_pool(pool_sample)

        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)
        print("update the support set in Offline mode...")
        self.retriever.update_offline()

        self.predictions = []
        self.test_sample_num = 0
        self.right_sample_num = 0
        print(" size of support set:",len(self.retriever.demonstrations))

        for idx in tqdm(shuffled_indices, desc=f"Inference ImageNet..."):
            self.inference(test_dataset[idx])

        
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc

        return results,self.predictions