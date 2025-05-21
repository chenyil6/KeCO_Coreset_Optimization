import argparse
from inferencers_mixup import *
import logging
from transformers import AutoTokenizer, CLIPModel,AutoProcessor, AutoModelForVision2Seq
import sys
sys.path.append('/path/to/KeCO_Coreset_Optimization')
from open_flamingo_v2.open_flamingo.src.factory import create_model_and_transforms
from transformers import IdeficsForVisionText2Text, AutoProcessor
import json
import logging
from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration


logging.getLogger("transformers").setLevel(logging.CRITICAL)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default="idefics_v2",
    )       
    parser.add_argument("--imagenet_root", type=str, default="/tmp")
    parser.add_argument("--result_folder", type=str, default="./result")
    parser.add_argument("--method", type=str, default="Offline_ICL")# FewShot;Online_ICL;Offline_ICL
    parser.add_argument("--seed", type=int, default=42)      
    parser.add_argument("--stream", type=int, default=4000)     
    parser.add_argument("--bank", type=str, default="initial") #initial; total
    parser.add_argument("--sample_method", type=str, default="random") # random;k_center_greedy;
    # Hyper parameters for OnlineICL
    parser.add_argument("--select_strategy", type=str, default="cosine")# cosine;l2;random
    parser.add_argument("--target_select", type=str, default="least_similarity") # random; least_similarity;most_similarity
    parser.add_argument("--dnum", type=int, default=2)
    parser.add_argument("--M", type=int, default=1000) #5*200
    parser.add_argument("--catergory_num", type=int, default=200) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--num", type=int, default=1000)  
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--online", type=str, default="fixed") 
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    device_set = "cuda:" + str(args.device)
    device = torch.device(device_set)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    if args.model == "open_flamingo_3b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="mpt/mpt-1b-redpajama-200b",
            tokenizer_path="mpt/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            precision="fp16",
            inference=True,
            device=device_set,
            checkpoint_path="/path/to/checkpoint.pt"
        )
    elif args.model == "idefics_v2":
        checkpoint = "idefics/idefics2-8b-base"
        model = AutoModelForVision2Seq.from_pretrained(checkpoint,local_files_only=True, torch_dtype=torch.bfloat16).to(device_set)
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(checkpoint, do_image_splitting=False, size={"longest_edge": 448, "shortest_edge": 378})
        #processor = AutoProcessor.from_pretrained(checkpoint)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
    elif args.model == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        processor = AutoProcessor.from_pretrained("qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if args.method == "Online_ICL":
        if args.model == "open_flamingo_3b":
            inferencer = Online_ICL(args, tokenizer, model, image_processor,device)
        elif args.model == "idefics_v2":
            inferencer = Online_ICL(args, tokenizer, model, image_processor,device,processor)
        elif args.model == "qwen2_vl":
            inferencer = Online_ICL(args, None, model, None,device,processor)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        results, predictions = inferencer.run()
    elif args.method == "FewShot":
        if args.model == "open_flamingo_3b":
            inferencer = FewShot(args, tokenizer, model, image_processor,  device)
        elif args.model == "idefics_v2":
            inferencer = FewShot(args, tokenizer, model, image_processor,device,processor)
        elif args.model == "qwen2_vl":
            inferencer = FewShot(args, None, model, None,device,processor)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        
        results, predictions = inferencer.run()
    elif args.method == "Offline_ICL":
        if  args.model == "open_flamingo_3b":
            inferencer = Offline_ICL(args, tokenizer, model, image_processor,  device,processor)
        elif args.model == "idefics_v2":
            inferencer = Offline_ICL(args, tokenizer, model, image_processor,device,processor)
        elif args.model == "qwen2_vl":
            inferencer = Offline_ICL(args, None, model, None,device,processor)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        
        results, predictions = inferencer.run()
    else:
        print("Method is invalid.")
        results = None

    results = {"device":args.device, "method": args.method, "select_strategy": args.select_strategy, "M": args.M,
               "batch_size":args.batch_size,"alpha":args.alpha,"catergory_num":args.catergory_num,"dnum":args.dnum,"target_select":args.target_select,
               "stream":args.stream,"bank":args.bank,"epoch":args.epoch,"num":args.num,"sample_method":args.sample_method,"results": results}
    print("-------------------------final-results-----------------------")
    print(results)
    
    if args.method == "Online_ICL": 
        res_file = os.path.join(args.result_folder, f"{args.method}-M={args.M}-select_strategy={args.select_strategy}-sample_method={args.sample_method}-"
                                                f"-alpha={args.alpha}-shot={args.dnum}-target_select={args.target_select}-stream={args.stream}.json")

    elif args.method == "FewShot":# FewShot
        res_file = os.path.join(args.result_folder, f"{args.method}-M={args.M}-select_strategy={args.select_strategy}-sample_method={args.sample_method}-"
                                                f"-shot={args.dnum}-bank={args.bank}.json")
    elif args.method == "Offline_ICL":
        res_file = os.path.join(args.result_folder, f"{args.dataset_mode}-{args.method}-M={args.M}-select_strategy={args.select_strategy}-sample_method={args.sample_method}-"
                                                f"-shot={args.dnum}-epoch={args.epoch}-num={args.num}.json")

        # load the prediction results to a json file
    
    with open(res_file, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
    
    print("save the prediction results to a json file:",res_file)

    