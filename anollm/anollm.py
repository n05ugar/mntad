'''
Original Copyright (c) 2022 Kathrin Seßler and Vadim Borisov. Licensed under the MIT License.
Part of code is adapted from the GReaT repository (https://github.com/kathrinse/be_great/tree/main)
Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''
import os
import warnings
import pickle

import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig
from torch.nn import CrossEntropyLoss
import typing as tp
from transformers import Trainer
from pathlib import Path

from anollm.anollm_trainer import AnoLLMTrainer, MNTTrainer
from anollm.anollm_utils import _array_to_dataframe
from anollm.anollm_dataset import AnoLLMDataset, AnoLLMDataCollator
from anollm.mnt_anollm import AnoLLMNumericalDataCollator

from safetensors.torch import save_model, load_model

class AnoLLM:
	"""Fine-tune a causal language model and score tabular anomalies, optionally with MNT."""

	def __init__(
		self,
		llm: str,
		experiment_dir: str = "models",
		batch_size: int = 8,
		efficient_finetuning: str = "",
		max_length_dict: tp.Optional[tp.Dict[str, int]] = None,
		textual_columns: tp.List[str] = [],
		random_init: bool = False,
		no_random_permutation: bool = False,
		use_mnt: bool = False,
		mnt_max_bins: int = 128,
		mnt_binning_strategy: str = "kmeans",
		mnt_bayesian_blocks_p0: float = 0.05,
		use_numerical_embedding: bool = False,
		use_numerical_scaling: bool = True,
		numerical_columns: tp.List[str] = [],
		direct_serialize_numerical_columns: tp.List[str] = [],
		**train_kwargs,
	):
		"""

		Args:
			llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
			experiment_dir:  Directory, where the training checkpoints will be saved
			batch_size: Batch size used for fine-tuning
			efficient_finetuning: if efficient_finetuning is 'lora', the model will be fine-tuned with LoRA
			max_length_dict: Dictionary that contains the maximum length of each textual features. 
			use_mnt: if True, use MNT for numerical features
			mnt_max_bins: Maximum number of bins for MNT discretization
			mnt_binning_strategy: Binning strategy for MNT discretization
			mnt_bayesian_blocks_p0: Initial false-alarm probability for Bayesian Blocks
			use_numerical_scaling: if True, multiply bin embeddings by numerical scales
			numerical_columns: List of numerical columns to apply MNT
			direct_serialize_numerical_columns: Numerical columns to serialize directly instead of MNT
			train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
			 see here the full list of all possible values
			 https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
		"""
		self.efficient_finetuning = efficient_finetuning
		self.llm = llm
		self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		
		self.use_mnt = use_mnt
		self.use_numerical_embedding = use_numerical_embedding
		self.use_numerical_scaling = use_numerical_scaling
		self.numerical_columns = numerical_columns
		self.direct_serialize_numerical_columns = direct_serialize_numerical_columns
		self.mnt_encoder = None
		self.numerical_scale_stats = {}
		self.numerical_scale_method = "robust_clip_linear"
		self.numerical_scale_clip = 1.0
		self.numerical_scale_factor = 0.5
		self.numerical_scale_tau = 2.0
		self.numerical_scale_eps = 1e-6
		self.numerical_scale_min = 0.5
		self.numerical_scale_max = 1.5
		
		if use_mnt:
			from anollm.mnt_anollm import MNTConfig, MNTEncoder
			self.mnt_config = MNTConfig(
				max_bins=mnt_max_bins,
				binning_strategy=mnt_binning_strategy,
				bayesian_blocks_p0=mnt_bayesian_blocks_p0,
			)
			self.mnt_encoder = MNTEncoder(self.mnt_config)
		
		self.new_tokens = []
		
		if not random_init:
			self.model = AutoModelForCausalLM.from_pretrained(self.llm, torch_dtype=torch.bfloat16)
		else:
			config = AutoConfig.from_pretrained(self.llm)
			self.model = AutoModelForCausalLM.from_config(config)

		if use_mnt and len(numerical_columns) > 0:
			from anollm.mnt_anollm import add_mnt_tokens_to_tokenizer
			self.tokenizer, mnt_tokens = add_mnt_tokens_to_tokenizer(
				self.tokenizer, self.mnt_config
			)
			self.new_tokens.extend(mnt_tokens)

		if len(self.new_tokens) > 0:
			self.model.resize_token_embeddings(len(self.tokenizer))
			print(f"Resized model embeddings to {len(self.tokenizer)} tokens")
		
		if use_numerical_embedding:
			from anollm.mnt_anollm import wrap_model_with_numerical_awareness
			mnt_config = self.mnt_config if use_mnt else None
			self.model = wrap_model_with_numerical_awareness(
				self.model,
				self.tokenizer,
				mnt_config,
				use_numerical_scaling=self.use_numerical_scaling,
			)
			print("Applied numerical-aware embeddings to model")
			
		if self.efficient_finetuning == "lora":
			try:
				from peft import (
					LoraConfig,
					get_peft_model,
				)
			except ImportError:
				raise ImportError(
					"This function requires the 'peft' package. Please install it with - pip install peft"
				)

			lora_config = LoraConfig(
				r=8, 
				lora_alpha=32,
				target_modules=[
					"q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
				],  # SmolLM projection modules.
				lora_dropout=0.1,
				bias="none",
				# Train newly added token embeddings and their output projection.
				modules_to_save=["embed_tokens", "lm_head"] if len(self.new_tokens) > 0 else None,
			)
			self.model = get_peft_model(self.model, lora_config)
			self.model.print_trainable_parameters()

		self.experiment_dir = experiment_dir
		self.batch_size = batch_size
		self.max_length_dict = max_length_dict
		self.textual_columns = textual_columns
		self.no_random_permutation = no_random_permutation
		self.train_hyperparameters = train_kwargs

	def _metadata_path(self, path: str) -> str:
		return f"{path}.metadata.pkl"

	def _fit_numerical_scale_stats(self, df: pd.DataFrame):
		"""Fit per-feature robust stats on training numerical columns."""
		stats = {}
		for col in self.numerical_columns:
			if col not in df.columns:
				continue
			s = pd.to_numeric(df[col], errors="coerce")
			s = s[np.isfinite(s)]
			if len(s) == 0:
				continue
			median = float(s.median())
			q1 = float(s.quantile(0.25))
			q3 = float(s.quantile(0.75))
			iqr = q3 - q1
			if not np.isfinite(iqr) or abs(iqr) < 1e-8:
				iqr = 1.0
			stats[col] = {"median": median, "iqr": float(iqr)}
		self.numerical_scale_stats = stats

	def _set_dataset_numerical_scale_stats(self, dataset: AnoLLMDataset):
		dataset.set_numerical_scale_stats(
			self.numerical_scale_stats,
			clip=self.numerical_scale_clip,
			factor=self.numerical_scale_factor,
			method=self.numerical_scale_method,
			tau=self.numerical_scale_tau,
			eps=self.numerical_scale_eps,
			min_scale=self.numerical_scale_min,
			max_scale=self.numerical_scale_max,
		)

	def _create_dataset(self, df: pd.DataFrame) -> AnoLLMDataset:
		dataset = AnoLLMDataset.from_pandas(df, preserve_index=False)
		dataset.set_tokenizer(self.tokenizer)
		dataset.set_textual_columns(self.textual_columns)
		if self.use_mnt and self.mnt_encoder is not None:
			dataset.set_mnt_encoder(self.mnt_encoder, self.numerical_columns)
		dataset.set_direct_serialize_numerical_columns(self.direct_serialize_numerical_columns)
		if self.use_numerical_embedding:
			dataset.enable_numerical_embedding(self.numerical_columns)
			self._set_dataset_numerical_scale_stats(dataset)
		if self.no_random_permutation:
			dataset.fix_column_order()
		return dataset

	def _get_data_collator(self):
		if self.use_numerical_embedding:
			return AnoLLMNumericalDataCollator(tokenizer=self.tokenizer, use_input_scales=True)
		return AnoLLMDataCollator(self.tokenizer)

	@staticmethod
	def _unwrap_model(model):
		if hasattr(model, "module"):
			return AnoLLM._unwrap_model(model.module)
		if hasattr(model, "_original_model"):
			return AnoLLM._unwrap_model(model._original_model)
		return model

	def fit(
		self,
		data: tp.Union[pd.DataFrame, np.ndarray],
		column_names: tp.Optional[tp.List[str]] = None,
		resume_from_checkpoint: tp.Union[bool, str] = False,
		use_wandb: bool = False,
		data_val: tp.Union[pd.DataFrame, np.ndarray] = None,
		label_val: np.ndarray = None,
		eval_steps: int = 400,
		processed_data_dir: str = None
		) -> Trainer:
		"""Fine-tune AnoLLM using tabular data.

		Args:
			data: Pandas DataFrame that contains the tabular data
			column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
			DataFrame, the value is ignored

		Returns:
			AnoLLM Trainer used for the fine-tuning process
		"""
		df = _array_to_dataframe(data, columns=column_names)

		if self.use_mnt and len(self.numerical_columns) > 0:
			logging.info("Fitting MNT encoder on numerical features...")
			numerical_df = df[self.numerical_columns].select_dtypes(include=[np.number])
			if len(numerical_df.columns) > 0:
				self.mnt_encoder.fit(numerical_df)
				logging.info(f"MNT encoder fitted on {len(numerical_df.columns)} numerical features")

		if self.use_numerical_embedding and len(self.numerical_columns) > 0:
			self._fit_numerical_scale_stats(df)
			logging.info(f"Fitted robust numerical scale stats for {len(self.numerical_scale_stats)} features")

		logging.info("Convert data into HuggingFace dataset object...")
		dataset = self._create_dataset(df)

		processed_data_path = Path(processed_data_dir) / "train_data.pkl" if processed_data_dir is not None else None 
		dataset.prepare(is_eval = False, max_length_dict=self.max_length_dict, 
				  data_path=processed_data_path)
		print("Data 0:", self.tokenizer.decode(dataset[0]['input_ids'] ))
		logging.info("Create AnoLLM Trainer...")
		trainer_args = {}

		if data_val is not None:
			df_val = _array_to_dataframe(data_val, columns=column_names)
			dataset_val = self._create_dataset(df_val)
			dataset_val.set_anomaly_label(label_val)
			
			processed_data_path = Path(processed_data_dir) / "val_data.pkl" if processed_data_dir is not None else None 
			dataset_val.prepare(is_eval = True, max_length_dict=self.max_length_dict, 
					   data_path = processed_data_path)

			self.train_hyperparameters["eval_strategy"] = "steps"
			self.train_hyperparameters["eval_steps"] = eval_steps
			trainer_args["eval_dataset"] = dataset_val
		
		if use_wandb:
			self.train_hyperparameters["report_to"] = ["wandb"]
			self.train_hyperparameters["logging_strategy"] = "steps"
			self.train_hyperparameters["logging_dir"] = "./logs"
			self.train_hyperparameters["logging_steps"] = 50
			self.train_hyperparameters["log_level"] = 'info'	
		else:
			self.train_hyperparameters.setdefault("report_to", "none")
		
		training_args = TrainingArguments(
			self.experiment_dir,
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size * 2,
			save_strategy = 'no',
			max_grad_norm = 0.7,
			ddp_find_unused_parameters=True,
			do_eval=False,  # Avoid duplicating the scheduled evaluation at training end.
			**self.train_hyperparameters,
		)

		data_collator = self._get_data_collator()
		
		trainer_class = MNTTrainer if self.use_mnt else AnoLLMTrainer
		trainer = trainer_class(
			self.model,
			training_args,
			train_dataset=dataset,
			tokenizer=self.tokenizer,
			data_collator=data_collator,
			**trainer_args,
		)

		if data_val is not None:
			trainer.set_eval_setting(n_permutations=1)

		logging.info("Start training...")
		trainer.train(resume_from_checkpoint=resume_from_checkpoint)

		return trainer
	
	def decision_function(
		self, 
		df_test: pd.DataFrame,
		n_permutations: int = 16, 
		batch_size: int = 32,
		device: str = "cuda",
		feature_wise: bool = False,
		) -> np.ndarray:
		"""Score test samples using next-token NLL across column permutations.

		Returns an array of shape (n_test, n_permutations), or
		(n_test, n_features, n_permutations) when feature_wise is True.
		"""
		logging.info("Convert data into HuggingFace dataset object...")
		dataset = self._create_dataset(df_test)
		
		dataset.prepare(is_eval = True, max_length_dict=self.max_length_dict)
		
		collate_fn = self._get_data_collator()
		
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = False, 
												 collate_fn = collate_fn)
		
		self.model.to(device)
		comma_id =  self.tokenizer.convert_tokens_to_ids(',')
		n_col = len(df_test.columns)
		column_names = dataset.get_column_names()
		if feature_wise:
			anomaly_scores = np.zeros((len(df_test), n_col, n_permutations))
		else:
			anomaly_scores = np.zeros((len(df_test), n_permutations))

		loss_fct = CrossEntropyLoss(reduction="none")


		for perm_idx in tqdm(range(n_permutations)):
			start_idx = 0
			dataset.shuffle_column_order()
			for data in dataloader:
				encoded_batch = data["input_ids"].to(device)
				attn_mask = data["attention_mask"].to(device)
				end_idx = start_idx + len(encoded_batch)
				labels = encoded_batch 
				
				start_pos_batch = data["feature_value_start"]
				end_pos_batch = data["feature_value_end"]
				col_indices_batch = data["col_indices"]

				with torch.no_grad():
					model_inputs = {"input_ids": encoded_batch, "attention_mask": attn_mask}
					if self.use_numerical_embedding and "input_scales" in data:
						model_inputs["input_scales"] = data["input_scales"].to(device)
					out_logits = self.model(**model_inputs).logits

				# Logits at position t predict the label at t + 1.
				shift_logits = out_logits[..., :-1, :].contiguous()
				shift_labels = labels[..., 1:].contiguous()
				shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

				if feature_wise:
					score_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).cpu().to(torch.float32).numpy()

					for i in range(len(encoded_batch)):
						for j in range(n_col): 
							start_pos = start_pos_batch[i][j]
							end_pos = end_pos_batch[i][j]
							col_idx = col_indices_batch[i][j]
							anomaly_scores[start_idx+i, col_idx, perm_idx] = score_batch[i, start_pos:end_pos].sum()
				elif len(self.textual_columns) > 0:
					score_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).cpu().to(torch.float32).numpy()
					for i in range(len(encoded_batch)):
						score_single = 0
						for j in range(n_col): 
							start_pos = start_pos_batch[i][j]
							end_pos = end_pos_batch[i][j]
							col_idx = col_indices_batch[i][j]
							if column_names[col_idx] in self.textual_columns:
								score_single += score_batch[i, start_pos:end_pos].sum() / (end_pos - start_pos)
							else:
								score_single += score_batch[i, start_pos:end_pos].sum()
						anomaly_scores[start_idx+i, perm_idx] = score_single
				else:
					score_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).to(torch.float32)
					score_batch_sum = score_batch.sum(1)
					anomaly_scores[start_idx:end_idx, perm_idx] = score_batch_sum.cpu().numpy()
				start_idx = end_idx

		return anomaly_scores
	
	def save_state_dict(self, path: str):
		"""Save model weights and preprocessing metadata, unwrapping model wrappers."""
		directory = os.path.dirname(path)
		if os.path.isdir(directory):
			warnings.warn(f"Directory {path} already exists and is overwritten now.")
		else:
			os.mkdir(directory)

		model_to_save = self._unwrap_model(self.model)
		save_model(model_to_save, path)
		metadata = {
			"mnt_encoder": self.mnt_encoder,
			"numerical_scale_stats": self.numerical_scale_stats,
			"use_numerical_scaling": self.use_numerical_scaling,
			"numerical_scale_method": self.numerical_scale_method,
			"numerical_scale_clip": self.numerical_scale_clip,
			"numerical_scale_factor": self.numerical_scale_factor,
			"numerical_scale_tau": self.numerical_scale_tau,
			"numerical_scale_eps": self.numerical_scale_eps,
			"numerical_scale_min": self.numerical_scale_min,
			"numerical_scale_max": self.numerical_scale_max,
		}
		with open(self._metadata_path(path), "wb") as f:
			pickle.dump(metadata, f)
	
	def load_from_state_dict(self, path: str):
		"""Load AnoLLM model from state_dict

		Args:
			path: path where AnoLLM model is saved
		"""
		model_to_load = self._unwrap_model(self.model)
		load_model(model_to_load, path)
		metadata_path = self._metadata_path(path)
		if os.path.exists(metadata_path):
			with open(metadata_path, "rb") as f:
				metadata = pickle.load(f)
			self.mnt_encoder = metadata.get("mnt_encoder", self.mnt_encoder)
			self.numerical_scale_stats = metadata.get("numerical_scale_stats", {})
			self.use_numerical_scaling = metadata.get(
				"use_numerical_scaling", self.use_numerical_scaling
			)
			self.numerical_scale_method = metadata.get("numerical_scale_method", "robust_clip_linear")
			self.numerical_scale_clip = metadata.get("numerical_scale_clip", 3.0)
			self.numerical_scale_factor = metadata.get("numerical_scale_factor", 0.1)
			self.numerical_scale_tau = metadata.get("numerical_scale_tau", 2.0)
			self.numerical_scale_eps = metadata.get("numerical_scale_eps", 1e-6)
			self.numerical_scale_min = metadata.get("numerical_scale_min", None)
			self.numerical_scale_max = metadata.get("numerical_scale_max", None)
			embedding_wrapper = getattr(self.model, "_embedding_wrapper", None)
			if embedding_wrapper is not None:
				embedding_wrapper.use_numerical_scaling = self.use_numerical_scaling
