import random
import typing as tp
import os 
import math

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
MAX_COL_LENGTH = 128

class AnoLLMDataset(Dataset):
	"""Serialize tabular rows with column permutations and optional numerical scales."""

	def set_tokenizer(self, tokenizer):
		"""Set the Tokenizer

		Args:
			tokenizer: Tokenizer from HuggingFace
		"""
		self.tokenizer = tokenizer 
	
	def set_anomaly_label(self, labels):
		assert len(labels) == len(self._data)
		self.anomaly_labels = labels

	def set_textual_columns(self, columns: tp.List[str]):
		col_list = self.get_column_names()
		for col in columns:
			if col not in col_list:
				raise ValueError("Column {} not in the dataset.".format(col))
		self.textual_columns = columns
	
	def set_mnt_encoder(self, mnt_encoder, numerical_columns: tp.List[str]):
		"""Set MNT encoder for numerical features
		
		Args:
			mnt_encoder: MNTEncoder instance
			numerical_columns: List of numerical column names
		"""
		self.mnt_encoder = mnt_encoder
		self.numerical_columns = numerical_columns

	def set_direct_serialize_numerical_columns(self, columns: tp.List[str]):
		"""Set numerical columns that should bypass MNT and be string-formatted directly."""
		self.direct_serialize_numerical_columns = columns
	
	def enable_numerical_embedding(self, numerical_columns: tp.List[str]):
		"""Enable numerical embedding support
		
		Args:
			numerical_columns: List of numerical column names
		"""
		self.use_numerical_embedding = True
		self.numerical_columns = getattr(self, 'numerical_columns', []) or numerical_columns
	
	def enable_numerical_embeddings(self, use_numerical_embedding: bool = True):
		"""Enable numerical embedding features
		
		Args:
			use_numerical_embedding: Whether to enable numerical embeddings
		"""
		self.use_numerical_embedding = use_numerical_embedding

	def set_numerical_scale_stats(
		self,
		stats: tp.Dict[str, tp.Dict[str, float]],
		clip: float = 1.0,
		factor: float = 0.5,
		method: str = "robust_clip_linear",
		tau: float = 2.0,
		eps: float = 1e-6,
		min_scale: tp.Optional[float] = 0.5,
		max_scale: tp.Optional[float] = 1.5,
	):
		"""Set per-feature robust normalization stats for numerical input scales."""
		self.numerical_scale_stats = stats or {}
		self.numerical_scale_clip = clip
		self.numerical_scale_factor = factor
		self.numerical_scale_method = method
		self.numerical_scale_tau = tau
		self.numerical_scale_eps = eps
		self.numerical_scale_min = min_scale
		self.numerical_scale_max = max_scale

	def _clamp_numerical_scale(self, scale: float) -> float:
		min_scale = getattr(self, 'numerical_scale_min', 0.5)
		max_scale = getattr(self, 'numerical_scale_max', 1.5)
		if min_scale is not None:
			scale = max(float(min_scale), scale)
		if max_scale is not None:
			scale = min(float(max_scale), scale)
		return scale

	def _get_numerical_scale_value(self, column_name: str, raw_value) -> float:
		"""Use fitted robust statistics for scaling, or retain the raw value if absent."""
		try:
			value = float(raw_value) if raw_value is not None else 1.0
		except (ValueError, TypeError):
			return 1.0

		stats = getattr(self, 'numerical_scale_stats', {}).get(column_name)
		if not stats:
			return value

		eps = float(getattr(self, 'numerical_scale_eps', 1e-6))
		median = float(stats.get('median', 0.0))
		iqr = float(stats.get('iqr', 1.0))
		denominator = iqr + eps
		if not math.isfinite(denominator) or denominator == 0:
			denominator = eps if eps != 0 else 1e-6
		z_value = (value - median) / denominator

		factor = getattr(self, 'numerical_scale_factor', 0.2)
		method = getattr(self, 'numerical_scale_method', 'robust_clip_linear')
		if method == "robust_tanh":
			tau = float(getattr(self, 'numerical_scale_tau', 2.0))
			if not math.isfinite(tau) or tau == 0:
				tau = 2.0
			return self._clamp_numerical_scale(1.0 + factor * math.tanh(z_value / tau))
		if method == "robust_clip_linear":
			clip = getattr(self, 'numerical_scale_clip', 1.0)
			if clip is not None:
				z_value = max(-clip, min(clip, z_value))
			return self._clamp_numerical_scale(1.0 + factor * z_value)
		raise ValueError(f"Unknown numerical scale method: {method}")
	
	def get_n_columns(self):
		row = self._data.fast_slice(0, 1)
		return row.num_columns

	def get_column_names(self):
		row = self._data.fast_slice(0, 1)
		return row.column_names
	
	def shuffle_column_order(self):
		"""Use one shuffled column order for all rows in an evaluation pass."""
		row = self._data.fast_slice(0, 1)
		self.shuffle_idx = list(range(row.num_columns))
		random.shuffle(self.shuffle_idx)
	
	def fix_column_order(self):
		row = self._data.fast_slice(0, 1)
		self.shuffle_idx = list(range(row.num_columns))
	
	def prepare(
		self,
		is_eval: bool = True, 
		max_length_dict: tp.Optional[tp.Dict[str, int]] = {},
		data_path = None,
		):
		"""Tokenize feature names and values, optionally reusing a cached tokenization.

		On a cache miss, truncate each column to its configured length or MAX_COL_LENGTH.
		"""
		self.is_eval = is_eval
		n_col = self.get_n_columns()
		column_names = self.get_column_names()
		self.processed_data = [] 
		self.tokenized_feature_names = []
		bos_token_id = self.tokenizer.bos_token_id
		
		for col_idx in range(n_col):
			feature_names = ' ' + column_names[col_idx] + ' '
			tokenized_feature_names = self.tokenizer(feature_names)
			tokenized_is = self.tokenizer('is ')
			if bos_token_id and tokenized_feature_names['input_ids'][0] == bos_token_id:
				tokenized_feature_names['input_ids'] = tokenized_feature_names['input_ids'][1:]
				tokenized_is['input_ids'] = tokenized_is['input_ids'][1:]

			self.tokenized_feature_names.append(tokenized_feature_names["input_ids"] + tokenized_is["input_ids"])
		
		if data_path is not None and os.path.exists(data_path):
			self.processed_data = pkl.load(open(data_path, 'rb'))
		else:
			for key in tqdm(range(len(self._data))):
				row = self._data.fast_slice(key, 1)
				tokenized_texts = []
				for col_idx in range(n_col):
					column_name = column_names[col_idx]
					raw_value = row.columns[col_idx].to_pylist()[0]
					feature_values = str(raw_value).strip()
					if len(feature_values) == 0:
						feature_values = "None"

					direct_serialize_columns = getattr(self, 'direct_serialize_numerical_columns', [])
					if column_name in direct_serialize_columns and raw_value is not None:
						try:
							feature_values = f"{float(raw_value):.1f}"
						except (TypeError, ValueError):
							pass
					
					if hasattr(self, 'mnt_encoder') and self.mnt_encoder is not None and hasattr(self, 'numerical_columns'):
						if column_name in self.numerical_columns:
							try:
								numeric_value = float(feature_values) if feature_values != "None" else 0.0
								if column_name in self.mnt_encoder.bin_edges:
									token_id = self.mnt_encoder.transform_value(column_name, numeric_value)
									bin_index = token_id - self.mnt_encoder.config.start_token_id
									if 0 <= bin_index < self.mnt_encoder.config.max_bins:
										feature_values = f"{self.mnt_encoder.config.mnt_token_prefix}{bin_index}"
							except Exception as e:
								# Keep the serialized value if MNT conversion fails.
								pass
					
					data = self.tokenizer(feature_values)
					if bos_token_id and data['input_ids'][0] == bos_token_id:
						data['input_ids'] = data['input_ids'][1:]

					tokenized_texts.append(data["input_ids"])
					if len(data["input_ids"]) == 0:
						print("Warning: tokenized text is empty.", column_names[col_idx],len( feature_values),feature_values)
				self.processed_data.append(tokenized_texts)
			
			for col_idx in range(n_col):
				name = column_names[col_idx]
				if name not in max_length_dict:
					max_length = MAX_COL_LENGTH
				else:
					max_length = max_length_dict[name]
				assert isinstance(max_length, int)
				
				for data_idx in range(len(self.processed_data)):
					length = len(self.processed_data[data_idx][col_idx]) + len(self.tokenized_feature_names[col_idx])
					if length >= max_length:
						self.processed_data[data_idx][col_idx] = self.processed_data[data_idx][col_idx][:max_length - len(self.tokenized_feature_names[col_idx])]
			if data_path is not None:
				pkl.dump(self.processed_data, open(data_path, 'wb'))
		print("Preprocessing done.")

	def _getitem(
		self, 
		key: tp.Union[int, slice, str], 
		decoded: bool = True, 
		**kwargs
	) -> tp.Union[tp.Dict, tp.List]:
		"""
		Get one instance of the tabular data, permuted, converted to text and tokenized.
		"""
		row = self._data.fast_slice(key, 1)
		

		if "shuffle_idx" in self.__dict__: 
			shuffle_idx = self.shuffle_idx
		else:
			shuffle_idx = list(range(row.num_columns))
			random.shuffle(shuffle_idx)
		
		comma_id =  self.tokenizer.convert_tokens_to_ids(',')
		eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
		bos_token_id = self.tokenizer.bos_token_id
		if self.is_eval:
			tokenized_text = {"input_ids": [], "attention_mask": [], "feature_value_start":[],
							"feature_value_end":[],'col_indices':shuffle_idx}
		else:
			tokenized_text = {"input_ids": [], "attention_mask": []}
		
		if getattr(self, 'use_numerical_embedding', False):
			tokenized_text["input_scales"] = []
			tokenized_text["token_type_ids"] = []
			
		if bos_token_id:
			tokenized_text["input_ids"] = [bos_token_id]
			if getattr(self, 'use_numerical_embedding', False):
				tokenized_text["input_scales"].append(1.0)
				tokenized_text["token_type_ids"].append(0)

		if hasattr(self, "processed_data"):
			start_idx = 0
			for idx, col_idx in enumerate(shuffle_idx):
				tokenized_feature_names = self.tokenized_feature_names[col_idx]
				tokenized_feature_values = self.processed_data[key][col_idx]
				tokenized_col = tokenized_feature_names + tokenized_feature_values 
				
				if idx == len(shuffle_idx) - 1:
					tokenized_text["input_ids"] += tokenized_col + [eos_id]
				else:
					tokenized_text["input_ids"] += tokenized_col + [comma_id]
				
				if getattr(self, 'use_numerical_embedding', False):
					column_name = self._data.column_names[col_idx]
					is_numerical = column_name in getattr(self, 'numerical_columns', [])
					
					# Names and separators carry neutral scales; values use feature statistics.
					for _ in tokenized_feature_names:
						tokenized_text["input_scales"].append(1.0)
						tokenized_text["token_type_ids"].append(0)
					
					for _ in tokenized_feature_values:
						if is_numerical:
							raw_value = row.columns[col_idx].to_pylist()[0]
							scale_value = self._get_numerical_scale_value(column_name, raw_value)
							tokenized_text["input_scales"].append(scale_value)
							tokenized_text["token_type_ids"].append(1)
						else:
							tokenized_text["input_scales"].append(1.0)
							tokenized_text["token_type_ids"].append(0)
					
					if idx == len(shuffle_idx) - 1:
						tokenized_text["input_scales"].append(1.0)
						tokenized_text["token_type_ids"].append(0)
					else:
						tokenized_text["input_scales"].append(1.0)
						tokenized_text["token_type_ids"].append(0)
				
				if self.is_eval:
					tokenized_text["feature_value_start"].append(start_idx + len(tokenized_feature_names) -1 )
					tokenized_text["feature_value_end"].append(start_idx + len(tokenized_col) )
				start_idx += len(tokenized_col) + 1
		else:
			raise ValueError("processed_data is not found. Please run prepare function first.")	
		tokenized_text["attention_mask"] += [1] * len(tokenized_text["input_ids"])
		
		if getattr(self, 'use_numerical_embedding', False):
			if len(tokenized_text["input_scales"]) != len(tokenized_text["input_ids"]):
				raise ValueError("input_scales length must match input_ids length")
			if len(tokenized_text["token_type_ids"]) != len(tokenized_text["input_ids"]):
				raise ValueError("token_type_ids length must match input_ids length")
		
		return tokenized_text
	
	def get_item_test(self, key):
		row = self._data.fast_slice(key, 1)
		shuffle_idx = list(range(row.num_columns))
		random.shuffle(shuffle_idx)
		
		shuffled_text = ",".join(
			[
				" %s is %s "
				% (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip() )
				for i in shuffle_idx
			]
		)
		tokenized_text = self.tokenizer(shuffled_text, padding=True)

		return shuffled_text, tokenized_text 
	
	def __getitems__(self, keys: tp.Union[int, slice, str, list]):
		if isinstance(keys, list):
			return [self._getitem(key) for key in keys]
		else:
			return self._getitem(keys)

@dataclass
class AnoLLMDataCollator(DataCollatorWithPadding):
	"""Create causal-LM labels from padded input IDs."""

	def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
		batch = self.tokenizer.pad(
			features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=self.return_tensors,
		)
		batch["labels"] = batch["input_ids"].clone()
		return batch

class AnoLLMDataLoader(DataLoader):
	'''
	Add set_epoch function so that huggingface trainer can call it 
	'''
	def set_epoch(self, epoch):
		if hasattr(self.sampler, "set_epoch"):
			self.sampler.set_epoch(epoch)
			print("Set epoch", epoch)
