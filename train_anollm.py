import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import argparse
import torch.distributed as dist
import torch
import wandb
import time
from transformers import set_seed

from anollm import AnoLLM
from src.data_utils import load_data, DATA_MAP, resolve_mnt_numerical_columns, get_text_columns, get_max_length_dict

MODEL_IDS = {
    "smol": "HuggingFaceTB/SmolLM-135M",
    "smol-360": "HuggingFaceTB/SmolLM-360M",
    "smol-1.7b": "HuggingFaceTB/SmolLM-1.7B",
}
MODEL_TAGS = {
    MODEL_IDS["smol"]: "smolLM",
    MODEL_IDS["smol-360"]: "smolLM360",
    MODEL_IDS["smol-1.7b"]: "smolLM1.7B",
}


def add_shared_args(parser):
    parser.add_argument("--dataset", choices=[name.lower() for name in DATA_MAP], default="wine")
    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--setting", choices=("semi_supervised", "unsupervised"),
                        default="semi_supervised")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--binning",
                        choices=("none", "standard", "quantile", "equal_width", "language"),
                        default="none")
    parser.add_argument("--n_buckets", type=int, default=10)
    parser.add_argument("--numerical_preprocessing", choices=("yj", "standard", "none"),
                        default="yj")

    parser.add_argument("--use_mnt", action="store_true")
    parser.add_argument("--uniform_mnt", action="store_true")
    parser.add_argument("--mnt_max_bins", type=int, default=128)
    parser.add_argument("--mnt_binning_strategy",
                        choices=("kmeans", "quantile", "uniform", "ckmeans_1d_dp",
                                 "bayesian_blocks"),
                        default="kmeans")
    parser.add_argument("--mnt_bayesian_blocks_p0", type=float, default=0.05)
    parser.add_argument("--numerical_scaling", choices=("enabled", "disabled"),
                        default="enabled")

    parser.add_argument("--model", choices=("gpt2", "distilgpt2", *MODEL_IDS), default="smol")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--no_random_permutation", action="store_true")
    return parser


def finalize_shared_args(parser, args):
    if args.mnt_max_bins < 1:
        parser.error("--mnt_max_bins must be positive")
    if args.n_buckets < 1:
        parser.error("--n_buckets must be positive")
    if args.use_mnt and args.binning != "none":
        parser.error("--binning must be none when --use_mnt is enabled")
    if args.uniform_mnt and not args.use_mnt:
        parser.error("--uniform_mnt requires --use_mnt")
    if not 0 < args.mnt_bayesian_blocks_p0 < 1:
        parser.error("--mnt_bayesian_blocks_p0 must be between 0 and 1")
    if args.exp_dir is None:
        args.exp_dir = (Path("exp") / args.dataset / args.setting
                        / f"split{args.n_splits}" / f"split{args.split_idx}")
    args.model = MODEL_IDS.get(args.model, args.model)
    return args


def get_args():
    parser = add_shared_args(argparse.ArgumentParser())
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project", default="AnoLLM")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=500)

    args = finalize_shared_args(parser, parser.parse_args())
    args.save_dir = args.exp_dir / "models"
    os.makedirs(args.save_dir, exist_ok=True)
    return args

def get_run_name(args):
    parts = ["anollm", f"lr{args.lr}", args.binning,
             MODEL_TAGS.get(args.model, args.model)]

    if args.binning != "none" and args.n_buckets != 10:
        parts.append(f"b{args.n_buckets}")

    if args.use_mnt:
        parts.extend((f"bin{args.mnt_max_bins}", args.mnt_binning_strategy,
                      args.numerical_preprocessing))
        if args.uniform_mnt:
            parts.append("allNumeric")
        if (args.mnt_binning_strategy == "bayesian_blocks"
                and args.mnt_bayesian_blocks_p0 != 0.05):
            parts.append(f"bbP{args.mnt_bayesian_blocks_p0}")
        if args.numerical_scaling == "disabled":
            parts.append("scaleOff")
    elif args.numerical_preprocessing != "yj":
        parts.append(args.numerical_preprocessing)

    if args.seed != 42:
        parts.append(f"seed{args.seed}")
    if args.random_init:
        parts.append("random_init")
    if args.no_random_permutation:
        parts.append("no_random_permutation")
    if args.lora:
        parts.append("lora")
    parts.append("test")
    return "_".join(parts)


def main():
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	args = get_args()
	set_seed(args.seed)
	if dist.get_rank() == 0:
		X_train, X_test, y_train, y_test = load_data(args)
	dist.barrier()
	if dist.get_rank() != 0:
		X_train, X_test, y_train, y_test = load_data(args)
	dist.barrier()
	
	run_name = get_run_name(args)
	efficient_finetuning = 'lora' if args.lora else ''
	model_path = args.save_dir / '{}.pt'.format(run_name)
	dataset_tmp_path = args.save_dir / (run_name + '_data')
	
	os.makedirs(dataset_tmp_path, exist_ok= True)
	print("Model path:", model_path)	
	if os.path.exists(model_path):
		print("Model exists, skip training")
		return

	max_length_dict = get_max_length_dict(args.dataset)
	text_columns = get_text_columns(args.dataset)
	
	numerical_columns = []
	direct_serialize_numerical_columns = []
	if args.use_mnt:
		numerical_columns, direct_serialize_numerical_columns = resolve_mnt_numerical_columns(
			X_train,
			args.dataset,
			uniform_mnt=args.uniform_mnt,
		)
		print(f"Detected numerical columns for MNT: {numerical_columns}")
	
	def get_model():
		model = AnoLLM(args.model,
					batch_size=args.batch_size,
					max_steps = args.max_steps,
					efficient_finetuning = efficient_finetuning,
					max_length_dict=max_length_dict, 
					textual_columns = text_columns,
					random_init=args.random_init,
					no_random_permutation=args.no_random_permutation,
					use_mnt=args.use_mnt,
					mnt_max_bins=args.mnt_max_bins,
					mnt_binning_strategy=args.mnt_binning_strategy,
					mnt_bayesian_blocks_p0=args.mnt_bayesian_blocks_p0,
					use_numerical_embedding=args.use_mnt,
					use_numerical_scaling=args.numerical_scaling == "enabled",
					numerical_columns=numerical_columns,
					direct_serialize_numerical_columns=direct_serialize_numerical_columns,
					bf16=True,
					adam_beta2=0.99,
					adam_epsilon=1e-7,
					learning_rate=args.lr,
					seed=args.seed,
				)
		return model 
	if dist.get_rank() == 0:
		anollm = get_model()
	dist.barrier()
	if dist.get_rank() != 0:
		anollm = get_model()
	dist.barrier()
	anollm.model.to(local_rank)
	if args.wandb and dist.get_rank() == 0: 
		run = wandb.init(
			entity=args.entity,
			project=args.project,
			name = "{}_splits{}_{}_{}".format(args.dataset, args.split_idx, args.n_splits, run_name),
		)
	if len(X_test) > 3000:
		np.random.seed(args.seed)
		X_test.reset_index(drop = True, inplace = True)
		indices = np.random.choice(len(X_test), 3000, replace = False)
		X_test = X_test.loc[indices].reset_index(drop = True)
		y_test = y_test[indices]
	if not args.wandb:
		X_test, y_test = None, None
	
	start_time = time.time()
	trainer = anollm.fit(X_train, X_train.columns.to_list(), 
					  use_wandb = args.wandb, 
					  data_val=X_test, 
					  label_val = y_test,
					  eval_steps = args.eval_steps,
					  processed_data_dir = dataset_tmp_path,
			)
	end_time = time.time()

	if dist.get_rank() == 0:
		
		print("Training time:", end_time - start_time)
		run_time_dir = args.exp_dir / "run_time" / "train"
		os.makedirs(run_time_dir, exist_ok = True)
		run_time_path = run_time_dir / "{}.txt".format(run_name)
		with open(run_time_path, 'w') as f:
			f.write(str(end_time - start_time))

		print("Save model to ", model_path)
		anollm.save_state_dict(model_path)
		
		
	dist.destroy_process_group()

if __name__ == "__main__":
	dist.init_process_group(backend="nccl") 
	main()
