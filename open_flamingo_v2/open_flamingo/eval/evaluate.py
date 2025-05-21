import argparse
import copy
import importlib
import json
import os
import random
import uuid
from collections import defaultdict

from einops import repeat
import more_itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import (
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
)
from tqdm import tqdm


from eval_datasets import VQADataset, ImageNetDataset
from classification_utils import (
    IMAGENET_CLASSNAMES,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
    HM_CLASSNAMES,
    HM_CLASS_ID_TO_LABEL,
)

from eval_model import BaseEvalModel

from ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.open_flamingo.src.flamingo import Flamingo
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from open_flamingo.open_flamingo.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
# Trial arguments
parser.add_argument("--shots", nargs="+", default=[4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Use key-value caching for classification evals to speed it up. Currently this doesn't underperforms for MPT models.",
)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to evaluate on VizWiz.",
)
parser.add_argument(
    "--eval_textvqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on TextVQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)
parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)
parser.add_argument(
    "--eval_hateful_memes",
    action="store_true",
    default=False,
    help="Whether to evaluate on Hateful Memes.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_karpathy_json_path",
    type=str,
    help="Path to the dataset_flickr30k.json file.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
)
## COCO Dataset
parser.add_argument(
    "--coco_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_val_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_karpathy_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_train_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_image_dir_path",
    type=str,
    help="Path to the vqav2/val2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_val2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_val2014_annotations.json file.",
    default=None,
)

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_image_dir_path",
    type=str,
    help="Path to the vizwiz test images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)

# TextVQA Dataset
parser.add_argument(
    "--textvqa_image_dir_path",
    type=str,
    help="Path to the textvqa images directory.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")

## Hateful Memes dataset
parser.add_argument(
    "--hateful_memes_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_test_annotations_json_path",
    type=str,
    default=None,
)

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo_v2.open_flamingo.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args, eval_model.device)
    # device_id = "cuda:" + str(model_args["device"])
    print(device_id)
    # device_id = torch.device(device_id)
    eval_model.set_device(device_id)
    # eval_model.init_distributed()

    # if args.model != "open_flamingo" and args.shots != [0]:
    #     raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_flickr30:
        print("Evaluating on Flickr30k...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="flickr",
                    min_generation_length=0,
                    max_generation_length=20,
                    num_beams=3,
                    style=True,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["flickr30"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_coco:
        print("Evaluating on COCO...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="coco",
                    style=True,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["coco"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                    scores.append(ok_vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
                results["ok_vqa"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            print("shot: ", shot)
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                    scores.append(vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
                results["vqav2"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                    scores.append(vizwiz_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
                results["vizwiz"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_textvqa:
        print("Evaluating on TextVQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                textvqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="textvqa",
                    max_generation_length=10,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} TextVQA score: {textvqa_score}")
                    scores.append(textvqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean TextVQA score: {np.nanmean(scores)}")
                results["textvqa"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_imagenet:
        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="imagenet",
                )
                if args.rank == 0:
                    print(
                        f"Shots {shot} Trial {trial} "
                        f"ImageNet score: {imagenet_score}"
                    )
                    scores.append(imagenet_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean ImageNet score: {np.nanmean(scores)}")
                results["imagenet"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_hateful_memes:
        print("Evaluating on Hateful Memes...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                hateful_memes_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="hateful_memes",
                )
                if args.rank == 0:
                    print(
                        f"Shots {shot} Trial {trial} "
                        f"Hateful Memes score: {hateful_memes_score}"
                    )
                    scores.append(hateful_memes_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean Hateful Memes score: {np.nanmean(scores)}")
                results["hateful_memes"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), len(train_dataset), replace=False)  #
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), len(test_dataset), replace=False) #
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader

def sample_batch_demos_from_query_set_caption(query_set, num_samples, batch, control_signals, seed, query_set_size):
    if not control_signals["clip"]:
        # query_set_list = get_query_set(query_set, query_set_size, seed)
        # output = [random.sample(query_set_list, num_samples) for _ in range(len(batch["image"]))]
        output = []
        for _ in range(len(batch["image"])):
            o = []
            a = random.sample(range(len(query_set)), num_samples)
            for j, i in enumerate(a):
                x = copy.deepcopy(query_set[i])
                o.append(x)
            output.append(o)
        return output
    elif control_signals["gold"]:
        return [[query_set.id2item(id) for id in batch["gold"][i][:num_samples]] for i in
                range(len(batch["image"]))]
    else:
        return [[query_set.id2item(id) for id in batch[control_signals["retrieval_type"]][i][:num_samples]] for i in
                range(len(batch["image"]))]

def sample_batch_demos_from_query_set(query_set, num_samples, batch, control_signals, seed, query_set_size):
    if control_signals["TR_TL"]:
        output = []
        for _ in range(len(batch["image"])):
            o = []
            a = random.sample(query_set.subSet_binary, num_samples)
            for j, i in enumerate(a):
                x = copy.deepcopy(query_set[i])
                o.append(x)
            output.append(o)
        return output
    if not control_signals["clip"]:
        # query_indice = np.random.choice(len(query_set), len(query_set), replace=False)
        # query_set = [query_set[i] for i in query_indice]
        query_set_list = get_query_set(query_set, query_set_size, seed)
        output = [random.sample(query_set_list, num_samples) for _ in range(len(batch["image"]))]
        if control_signals["SQAQAR_type"] == "sole":
            retrieval_icl = [[query_set.id2item(id)
                                     for id in batch[control_signals["retrieval_type"]][i][:num_samples]]
                                     for i in range(len(batch["image"]))]
            return [[retrieval_icl[i][idx] if idx == 0 else output[i][idx]
                              for idx in range(num_samples)]
                             for i in range(len(batch["image"]))]
        if control_signals["SQAQAR_type"] == "sole_wrong":
            retrieval_icl = [[query_set.id2item(id)
                                       for id in batch[control_signals["retrieval_type"]][i][:num_samples]]
                                      for i in range(len(batch["image"]))]
            batch_wrong = []
            for i in range(len(batch["image"])):
                for sample in retrieval_icl[i]:
                    if sample["answers"][0] != batch["answers"][i][0]:
                        batch_wrong.append(sample)
                        break
                    batch_wrong.append(sample)
                    break
            assert len(batch_wrong) == len(batch["image"])
            return [[batch_wrong[i] if idx == 0 else output[i][idx]
                 for idx in range(num_samples)]
                for i in range(len(batch["image"]))]
        return output
    else:
        if control_signals["retrieval_type"] == "mix_img_cap":
            return [[query_set.id2item(id) for id in batch["clip_images"][i][:(num_samples//2)] + batch["clip_captions"][i][:(num_samples//2)]] for i in range(len(batch["image"]))]
        else:
            return [[query_set.id2item(id) for id in batch[control_signals["retrieval_type"]][i][:num_samples]]
                    for i in range(len(batch["image"]))]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    style = False,
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
    elif dataset_name == "flickr":
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


    if style:
        control_signals = {
            "clip": False,
            "type": "romantic",  # humor; romantic; pos; neg
            "retrieval_type": "RS",
            "factual": True,
            "shots": num_shots,
            "instruction": True,
            "gold": False,
        }
        if control_signals["retrieval_type"] == "SIIR":
            control_signals["clip"] = True
        print(control_signals)
        train_dataset = StyleCaptionDataset(
            is_train=True,
            dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
            style_type=control_signals["type"],
        )

        test_dataset = StyleCaptionDataset(
            is_train=False,
            dataset_name=dataset_name,
            style_type=control_signals["type"],
        )
        random_uuid = "{}_{}_{}_{}_factual{}_instruction{}new_gold{}".format(dataset_name,
                                           control_signals["type"],
                                           control_signals["retrieval_type"],
                                           num_shots,
                                           control_signals["factual"],
                                           control_signals["instruction"],
                                           control_signals["gold"]
                                                  )
    else:
        train_dataset = CaptionDataset(
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=image_val_dir_path,
            annotations_path=annotations_path,
            is_train=True,
            dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
        )

        test_dataset = CaptionDataset(
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=image_val_dir_path,
            annotations_path=annotations_path,
            is_train=False,
            dataset_name=dataset_name,
        )
        random_uuid = "{}_".format(dataset_name)

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    # in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)



    # predictions = defaultdict()
    predictions = []

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    if not os.path.exists(f"{dataset_name}results_{random_uuid}.json"):
        for batch in tqdm(
            test_dataloader,
            desc=f"Running inference {dataset_name.upper()}",
            disable=args.rank != 0,
        ):
            batch_demo_samples = sample_batch_demos_from_query_set_caption(
                train_dataset, effective_num_shots, batch, control_signals, seed, args.query_set_size
            )

            batch_images = []
            batch_text = []
            for i in range(len(batch["image"])):
                if num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join(
                    [
                        eval_model.get_caption_prompt(caption=x["caption"].strip(),
                                                      factual_signal=control_signals["factual"],
                                                      factual=x["factual"][0].strip())
                        for x in batch_demo_samples[i]
                    ]
                )

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")
                if control_signals["instruction"]:
                    instruction_prompt = "According to the image, generate a negative-sentiment sentence. "
                    if control_signals["factual"]:
                        style_prompt_dic = {
                            "pos": "positive-sentiment",
                            "neg": "negative-sentiment",
                            "humor": "humorous style",
                            "romantic": "romantic style"
                        }
                        style_type = control_signals["type"]
                        instruction_prompt = f"According to the image, change the factual caption to " \
                                             f"{style_prompt_dic[style_type]}. "
                    context_text = instruction_prompt + context_text
                batch_text.append(context_text + eval_model.get_caption_prompt(factual_signal=control_signals["factual"],
                                                                               factual=batch["factual"][i][0].strip()))
            try:
                outputs = eval_model.get_outputs(
                    batch_images=batch_images,
                    batch_text=batch_text,
                    min_generation_length=min_generation_length,
                    max_generation_length=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            new_predictions = [
                postprocess_captioning_generation(out).replace('"', "") for out in outputs
            ]
            new_predictions = list(new_predictions)
            for i in range(len(batch["image"])):
                predictions.append({"caption": new_predictions[i],
                                    "image_id": batch["image_id"][i],
                                    "prompt_text": str(batch_text[i]),
                                    "prompt_images": [img['image_id'] for img in batch_demo_samples[i]],
                                    # "factual": batch["factual"][i],
                                    })

            # for i, sample_id in enumerate(batch["image_id"]):
            #     predictions[sample_id] = {
            #         "caption": new_predictions[i],
            #     }

        # all gather
        # all_predictions = [None] * args.world_size
        # torch.distributed.all_gather_object(all_predictions, predictions)  # list of dicts
        #
        # if args.rank != 0:
        #     return
        #
        # all_predictions = {
        #     k: v for d in all_predictions for k, v in d.items()
        # }  # merge dicts

        # save the predictions to a temporary file
        # results_path = f"{dataset_name}results_{uuid.uuid4()}.json"
        #
        # with open(results_path, "w") as f:
        #     f.write(
        #         json.dumps(
        #             [
        #                 {"image_id": k, "caption": all_predictions[k]["caption"]}
        #                 for k in all_predictions
        #             ],
        #             indent=4,
        #         )
        #     )
        all_predictions = predictions
        with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(all_predictions, indent=4))
    style_type = control_signals["type"]

    # delete the temporary file
    # os.remove(results_path)

    return metrics["CIDEr"] * 100.0


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    control_signals = {"clip": True,
                       # "model": "idefics",
                       "retrieval_type": "best_icd", # clip_images
                       "SQAQAR_": False,
                       "SQAQAR_type": "normal", # sole, sole_wrong, normal
                       "specification": False,
                       "order": "order",  # order; reverse
                       "mismatch": False,
                       "captions": False,
                       "TR_TL": False,
                       "TR_TL_type": "gold",  # gold; mismatch; new_mapping
                       "debug": "test_best_icd",
                       }
    print("control signals of prompt: ", control_signals)

    if control_signals["TR_TL"]:
        train_dataset = VQADataset(
            image_dir_path=train_image_dir_path,
            question_path=train_questions_json_path,
            annotations_path=train_annotations_json_path,
            is_train=True,
            dataset_name=dataset_name,
            signal=True, type=control_signals["TR_TL_type"]
        )
        test_dataset = VQADataset(
            image_dir_path=test_image_dir_path,
            question_path=test_questions_json_path,
            annotations_path=test_annotations_json_path,
            is_train=False,
            dataset_name=dataset_name,
            signal=True, type=control_signals["TR_TL_type"]
        )

    else:
        train_dataset = VQADataset(
            image_dir_path=train_image_dir_path,
            question_path=train_questions_json_path,
            annotations_path=train_annotations_json_path,
            is_train=True,
            dataset_name=dataset_name,
        )
        test_dataset = VQADataset(
            image_dir_path=test_image_dir_path,
            question_path=test_questions_json_path,
            annotations_path=test_annotations_json_path,
            is_train=False,
            dataset_name=dataset_name,
        )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    random_uuid = "{}_{}_spec{}_{}_{}".format(control_signals["retrieval_type"],
                                           num_shots,
                                           control_signals["specification"],
                                           control_signals["order"],
                                              control_signals["SQAQAR_type"])
    if control_signals["mismatch"]:
        random_uuid = "{}_spec{}_mismatch".format(num_shots,
                                               control_signals["specification"],
                                               )
    if control_signals["SQAQAR_"]:
        random_uuid += "_qa_plus"

    if control_signals["TR_TL"]:
        random_uuid = "test_TR_TL_{}_{}".format(num_shots, control_signals["TR_TL_type"])

    if control_signals["model"] == "idefics":
        random_uuid = "idefics_test{}_{}_{}_{}".format(
            args.num_samples, control_signals["retrieval_type"], num_shots, control_signals["debug"])
    # in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []

    def get_vqa_prompt(question, answer=None) -> str:
        if args.model == "open_flamingo":
            return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"
        if args.model == "idefics":
            fake_token = "<fake_token_around_image>"
            image_token = "<image>"
            icd_join_char = '\n'
            return f"{fake_token}{image_token}{fake_token}Question:{question} Short answer:{answer if answer is not None else ''}{icd_join_char if answer is not None else ''}"

    def build_prompts_with_context_caption(sample_image, sample_question, demo_samples, num_shots):
        fake_token = "<fake_token_around_image>"
        image_token = "<image>"
        icd_join_char = '\n'
        prompts = ["Instruction: provide an answer to the question. Use the image to answer.\n"]
        if num_shots > 0:
            for demo in demo_samples:
                prompt_part = [
                    # fake_token,
                    demo['image'],
                    # fake_token,
                    f"Question:{demo['question']} Short answer:{demo['answers'][0] if demo['answers'][0] is not None else ''}"
                    f"{icd_join_char if demo['answers'][0] is not None else ''}"
                ]
                prompts.extend(prompt_part)

        # add current test_sample
        # else:
        #     prompts = []
        current_prompt_part = [
            fake_token,
            sample_image,
            fake_token,
            f"Question:{sample_question} Short answer:"
        ]
        prompts.extend(current_prompt_part)
        return prompts

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    if not os.path.exists(f"{dataset_name}results_{random_uuid}.json"):
        for batch in tqdm(
            test_dataloader,
            desc=f"Running inference {dataset_name}",
            disable=args.rank != 0,
        ):
            batch_demo_samples = sample_batch_demos_from_query_set(
                train_dataset, effective_num_shots, batch,
                # control_signals["retrieval_type"], control_signals["clip"],
                control_signals,
                seed, args.query_set_size)

            batch_images = []
            batch_text = []
            if control_signals["model"] == "idefics":
                for i in range(len(batch["image"])):
                    if num_shots > 0:
                        context_prompt = build_prompts_with_context_caption(
                            sample_image=batch["image"][i],
                            sample_question=batch["question"][i],
                            demo_samples=batch_demo_samples[i],
                            num_shots=num_shots,
                            )  # get prompts
                    batch_text.append(context_prompt)
            else:
                for i in range(len(batch["image"])):
                    in_context_samples = batch_demo_samples[i]
                    if control_signals["order"] == "reverse":
                        in_context_samples.reverse()

                    if num_shots > 0:
                        context_images = [x["image"] for x in in_context_samples]
                    else:
                        context_images = []
                    batch_images.append(context_images + [batch["image"][i]])

                    if control_signals["mismatch"]:
                        from random import sample
                        in_context_samples = sample(in_context_samples, effective_num_shots)
                        context_text = "".join(
                            [
                                get_vqa_prompt(
                                    question=x["question"], answer=x["answers"][0]
                                )
                                for x in in_context_samples
                            ]
                        )
                    elif control_signals["SQAQAR_"] and control_signals["retrieval_type"] == "SQAQAR":
                        context_text = "".join(
                            [
                                get_vqa_prompt(
                                    question=x["QA_question"], answer=x["answers"][0]
                                )
                                for x in in_context_samples
                            ]
                        )
                    else:
                        context_text = "".join(
                            [
                                get_vqa_prompt(
                                    question=x["question"], answer=x["answers"][0]
                                )
                                for x in in_context_samples
                            ]
                        )
                    if control_signals["specification"]:
                        context_text = "According to the previous question and answer pair, " \
                                       "answer the final question. " + context_text
                    if args.model == "idefics":
                        context_text = "Instruction: provide an answer to the question. Use the image to answer.\n" + context_text

                    # Keep the text but remove the image tags for the zero-shot case
                    if num_shots == 0:
                        context_text = context_text.replace("<image>", "")

                    batch_text.append(
                        context_text + get_vqa_prompt(question=batch["question"][i])
                    )

            try:
                outputs = eval_model.get_outputs(
                    batch_images=batch_images,
                    batch_text=batch_text,
                    min_generation_length=min_generation_length,
                    max_generation_length=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception


            # process_function = (
            #     postprocess_ok_vqa_generation
            #     if dataset_name == "ok_vqa"
            #     else postprocess_vqa_generation
            # )
            #
            # new_predictions = map(process_function, outputs)
            def postprocess_vqa_generation(predictions):
                predictions = predictions.split("Short answer:")[-1].strip()
                predictions = predictions.split('\n')[0]
                return predictions
                # index_newline = predictions.find('\n')
                # index_dot = predictions.find('.')
                # if index_newline == -1 and index_dot == -1:
                #     return predictions
                # else:
                #     if index_newline == -1:
                #         index_newline = float('inf')
                #     if index_dot == -1:
                #         index_dot = float('inf')
                #     index_to_cut = min(index_newline, index_dot)
                #     if index_to_cut == index_dot:
                #         return predictions[:index_to_cut + 1]
                #     else:
                #         return predictions[:index_to_cut]

            new_predictions = [
                postprocess_vqa_generation(out).replace('"', "") for out in outputs
            ]

            # for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            #     predictions.append({"answer": new_prediction, "question_id": sample_id})
            new_predictions = list(new_predictions)
            for i in range(len(batch["image"])):
                predictions.append({"answer": new_predictions[i], "question_id": batch["question_id"][i],
                                    "prompt_text": str(batch_text[i]),
                                    "prompt_images": [img['image_id'] for img in batch_demo_samples[i]],
                                    "question": batch["question"][i]
                                    })

        # # all gather
        # all_predictions = [None] * args.world_size
        # torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
        # if args.rank != 0:
        #     return
        #
        # all_predictions = [
        #     item for sublist in all_predictions for item in sublist
        # ]  # flatten

        # save the predictions to a temporary file
        # random_uuid = str(uuid.uuid4())
        all_predictions = predictions
        with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc, _ = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        # os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        print("Temporary file saved to:", f"{dataset_name}results_{random_uuid}.json")
        acc = None

    return acc


def evaluate_classification(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    num_shots: int = 8,
    no_kv_caching=False,
    dataset_name: str = "imagenet",
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (str, optional): dataset name. Defaults to "imagenet".

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo "
            "models"
        )
    batch_size = args.batch_size
    num_samples = args.num_samples
    model, tokenizer = eval_model.model, eval_model.tokenizer

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "val"))
    elif dataset_name == "hateful_memes":
        train_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_train_annotations_json_path,
        )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
        seed,
    )

    acc1 = 0
    acc5 = 0

    if dataset_name == "imagenet":
        prompt_text = "<image>Output:"
    elif dataset_name == "hateful_memes":
        prompt_text = "<image>is an image with: '{meme_text}' written on it. Is it hateful? Answer: "

    predictions = []

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    for batch_idx, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        batch_images = []
        batch_text = []

        for idx in range(len(batch["image"])):
            # Choose a different set of random context samples for each sample
            # from the training set
            context_indices = np.random.choice(
                len(train_dataset), effective_num_shots, replace=False
            )

            in_context_samples = [train_dataset[i] for i in context_indices]

            if num_shots > 0:
                vision_x = [
                    eval_model.image_processor(data["image"]).unsqueeze(0)
                    for data in in_context_samples
                ]
            else:
                vision_x = []

            vision_x = vision_x + [
                eval_model.image_processor(batch["image"][idx]).unsqueeze(0)
            ]
            batch_images.append(torch.cat(vision_x, dim=0))

            def sample_to_prompt(sample):
                if dataset_name == "hateful_memes":
                    return prompt_text.replace("{meme_text}", sample["ocr"])
                else:
                    return prompt_text

            context_text = "".join(
                f"{sample_to_prompt(in_context_samples[i])}{in_context_samples[i]['class_name']}<|endofchunk|>"
                for i in range(effective_num_shots)
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text)

        # shape [B, T_img, C, h, w]
        vision_x = torch.stack(batch_images, dim=0)
        # shape [B, T_img, 1, C, h, w] where 1 is the frame dimension
        vision_x = vision_x.unsqueeze(2)

        # Cache the context text: tokenize context and prompt,
        # e.g. '<context> a picture of a '
        text_x = [
            context_text + sample_to_prompt({k: batch[k][idx] for k in batch.keys()})
            for idx, context_text in enumerate(batch_text)
        ]

        ctx_and_prompt_tokenized = tokenizer(
            text_x,
            return_tensors="pt",
            padding="longest",
            max_length=2000,
        )

        ctx_and_prompt_input_ids = ctx_and_prompt_tokenized["input_ids"].to(
            eval_model.device
        )
        ctx_and_prompt_attention_mask = (
            ctx_and_prompt_tokenized["attention_mask"].to(eval_model.device).bool()
        )

        def _detach_pkvs(pkvs):
            """Detach a set of past key values."""
            return list([tuple([x.detach() for x in inner]) for inner in pkvs])

        if not no_kv_caching:
            eval_model.cache_media(
                input_ids=ctx_and_prompt_input_ids,
                vision_x=vision_x.to(eval_model.device),
            )

            with torch.no_grad():
                precomputed = eval_model.model(
                    vision_x=None,
                    lang_x=ctx_and_prompt_input_ids,
                    attention_mask=ctx_and_prompt_attention_mask,
                    clear_conditioned_layers=False,
                    use_cache=True,
                )

            precomputed_pkvs = _detach_pkvs(precomputed.past_key_values)
            precomputed_logits = precomputed.logits.detach()

        else:
            precomputed_pkvs = None
            precomputed_logits = None

        if dataset_name == "imagenet":
            all_class_names = IMAGENET_CLASSNAMES
        else:
            all_class_names = HM_CLASSNAMES

        if dataset_name == "imagenet":
            class_id_to_name = IMAGENET_1K_CLASS_ID_TO_LABEL
        else:
            class_id_to_name = HM_CLASS_ID_TO_LABEL

        overall_probs = []
        for class_name in all_class_names:
            past_key_values = None
            # Tokenize only the class name and iteratively decode the model's
            # predictions for this class.
            classname_tokens = tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(eval_model.device)
            if classname_tokens.ndim == 1:  # Case: classname is only 1 token
                classname_tokens = torch.unsqueeze(classname_tokens, 1)

            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            if not no_kv_caching:
                # Compute the outputs one token at a time, using cached
                # activations.

                # Initialize the elementwise predictions with the last set of
                # logits from precomputed; this will correspond to the predicted
                # probability of the first position/token in the imagenet
                # classname. We will append the logits for each token to this
                # list (each element has shape [B, 1, vocab_size]).
                elementwise_logits = [precomputed_logits[:, -2:-1, :]]

                for token_idx in range(classname_tokens.shape[1]):
                    _lang_x = classname_tokens[:, token_idx].reshape((-1, 1))
                    outputs = eval_model.get_logits(
                        lang_x=_lang_x,
                        past_key_values=(
                            past_key_values if token_idx > 0 else precomputed_pkvs
                        ),
                        clear_conditioned_layers=False,
                    )
                    past_key_values = _detach_pkvs(outputs.past_key_values)
                    elementwise_logits.append(outputs.logits.detach())

                # logits/probs has shape [B, classname_tokens + 1, vocab_size]
                logits = torch.concat(elementwise_logits, 1)
                probs = torch.softmax(logits, dim=-1)

                # collect the probability of the generated token -- probability
                # at index 0 corresponds to the token at index 1.
                probs = probs[:, :-1, :]  # shape [B, classname_tokens, vocab_size]

                gen_probs = (
                    torch.gather(probs, 2, classname_tokens[:, :, None])
                    .squeeze(-1)
                    .cpu()
                )

                class_prob = torch.prod(gen_probs, 1).numpy()
            else:
                # Compute the outputs without using cached
                # activations.

                # contatenate the class name tokens to the end of the context
                # tokens
                _lang_x = torch.cat([ctx_and_prompt_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_and_prompt_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )

                outputs = eval_model.get_logits(
                    vision_x=vision_x.to(eval_model.device),
                    lang_x=_lang_x.to(eval_model.device),
                    attention_mask=_attention_mask.to(eval_model.device),
                    clear_conditioned_layers=True,
                )

                logits = outputs.logits.detach().float()
                probs = torch.softmax(logits, dim=-1)

                # get probability of the generated class name tokens
                gen_probs = probs[
                    :, ctx_and_prompt_input_ids.shape[1] - 1 : _lang_x.shape[1], :
                ]
                gen_probs = (
                    torch.gather(gen_probs, 2, classname_tokens[:, :, None])
                    .squeeze(-1)
                    .cpu()
                )
                class_prob = torch.prod(gen_probs, 1).numpy()

            overall_probs.append(class_prob)

        overall_probs = np.row_stack(overall_probs).T  # shape [B, num_classes]

        eval_model.uncache_media()

        def topk(probs_ary: np.ndarray, k: int) -> np.ndarray:
            """Return the indices of the top k elements in probs_ary."""
            return np.argsort(probs_ary)[::-1][:k]

        for i in range(len(batch_text)):
            highest_prob_idxs = topk(overall_probs[i], 5)

            top5 = [class_id_to_name[pred] for pred in highest_prob_idxs]

            y_i = batch["class_name"][i]
            acc5 += int(y_i in set(top5))
            acc1 += int(y_i == top5[0])

            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": top5[0],
                    "pred_score": overall_probs[i][highest_prob_idxs[0]]
                    if dataset_name == "hateful_memes"
                    else None,  # only for hateful memes
                }
            )

    # all gather
    all_predictions = [None] * args.world_size
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # Hack to remove samples with duplicate ids (only necessary for multi-GPU evaluation)
    all_predictions = {pred["id"]: pred for pred in all_predictions}.values()

    assert len(all_predictions) == len(test_dataset)  # sanity check

    if dataset_name == "hateful_memes":
        # return ROC-AUC score
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [pred["pred_score"] for pred in all_predictions]
        return roc_auc_score(gts, pred_scores)
    else:
        # return top-1 accuracy
        acc1 = sum(
            int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
        )
        return float(acc1) / len(all_predictions)


if __name__ == "__main__":
    main()
