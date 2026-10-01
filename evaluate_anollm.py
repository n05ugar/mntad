import os
import faulthandler

if __name__ == "__main__":
	faulthandler.enable(all_threads=True)

# AnoLLM evaluation is PyTorch-only. Prevent Transformers from importing
# TensorFlow and initializing a second CUDA runtime in the evaluator process.
os.environ.setdefault("USE_TF", "0")

import argparse

import numpy as np
import torch
import time

import torch.distributed as dist
from transformers import set_seed

from anollm import AnoLLM
from src.data_utils import load_data, resolve_mnt_numerical_columns, get_text_columns, get_max_length_dict
from train_anollm import add_shared_args, finalize_shared_args, get_run_name


def get_args():
    parser = add_shared_args(argparse.ArgumentParser())
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_permutations", type=int, default=100)
    return finalize_shared_args(parser, parser.parse_args())

def main():
	args = get_args()
	local_rank = int(os.environ.get("LOCAL_RANK", "0"))
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	rank = int(os.environ.get("RANK", "0"))
	if args.n_permutations < 1:
		raise ValueError("n_permutations must be positive")

	if not os.path.exists(args.exp_dir):
		raise ValueError("Experiment directory {} does not exist".format(args.exp_dir))
		
	score_dir = args.exp_dir / 'scores'
	run_name = get_run_name(args)

	score_path = score_dir / "{}.npy".format(run_name)
	print("score_path:", score_path, flush=True)
	if score_path.exists():
		print("Scores exist, skip evaluation", flush=True)
		return

	print(f"Evaluation device: cuda:{local_rank}, CUDA_VISIBLE_DEVICES="
	      f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}, world_size={world_size}", flush=True)
	torch.cuda.set_device(local_rank)
	# Single-GPU inference needs neither a process group nor DDP.
	if world_size > 1:
		dist.init_process_group(backend="nccl")
	try:
		set_seed(args.seed)
		_evaluate(args, local_rank, rank, world_size, score_path)
	finally:
		if world_size > 1 and dist.is_initialized():
			dist.destroy_process_group()


def _evaluate(args, local_rank, rank, world_size, score_path):
	run_name = score_path.stem
	score_dir = score_path.parent
	if rank == 0:
		os.makedirs(score_dir, exist_ok=True)

	remainder = args.n_permutations % world_size
	
	print("Loading evaluation data", flush=True)
	X_train, X_test, y_train, y_test = load_data(args)
	
	if not os.path.exists(score_path):
		model_dir = args.exp_dir / 'models'
		model_path = model_dir / '{}.pt'.format(run_name)
		
		efficient_finetuning = 'lora' if args.lora else ''
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
		
		print("Loading model and tokenizer", flush=True)
		model = AnoLLM(args.model,
							efficient_finetuning = efficient_finetuning,
							model_path = model_path,
							max_length_dict=max_length_dict, 
							textual_columns = text_columns,
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
					)
		print(text_columns, max_length_dict)
		
		print(f"Loading checkpoint: {model_path}", flush=True)
		model.load_from_state_dict(model_path)
		model.model.to(torch.device("cuda", local_rank))
		model.model.eval()
		# Each rank loads the same checkpoint and evaluates its own permutations.
		n_perm = args.n_permutations // world_size
		n_perm = n_perm + 1 if rank < remainder else n_perm

		print(f"Computing scores: rank={rank}, permutations={n_perm}", flush=True)
		start_time = time.time()	
		scores = model.decision_function(X_test, 
										n_permutations = n_perm, 
										batch_size = args.batch_size, 
										device = "cuda",
		)
		end_time = time.time()

		if world_size > 1:
			all_scores = [None for _ in range(world_size)]
			dist.all_gather_object(all_scores, scores)
		else:
			all_scores = [scores]

		if rank == 0:
			
			print("Inference time:", end_time - start_time, flush=True)
			
			run_time_dir = args.exp_dir / "run_time" / "test"
			os.makedirs(run_time_dir, exist_ok = True)
			run_time_path = run_time_dir / "{}.txt".format(run_name)
			with open(run_time_path, 'w') as f:
				f.write(str(end_time - start_time))
			
			all_scores = np.concatenate(all_scores, axis = 1)
			mean_scores = np.mean(all_scores, axis = 1)
			np.save(score_path, mean_scores)
			raw_score_path =  score_dir / "raw_{}.npy".format(run_name) 
			np.save(raw_score_path, all_scores)
	
if __name__ == '__main__':
	main()
