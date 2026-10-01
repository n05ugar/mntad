"""
AnoLLM with MNT support for SmolLM.

Components:
- NumericalAwareEmbedding: Embeddings with magnitude scaling for numerical tokens
- MNTEncoder: Configurable numerical discretization (bin tokens)
"""

import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import dataclass

from transformers.models.llama.modeling_llama import LlamaRMSNorm

logger = logging.getLogger(__name__)


def build_bin_token_set(tokenizer, prefix: str = "bin") -> Set[int]:
    """Build set of bin token IDs from tokenizer vocabulary."""
    bin_token_ids: Set[int] = set()
    if tokenizer is None:
        return bin_token_ids
    try:
        vocab = tokenizer.get_vocab()
        prefix_len = len(prefix)
        for token, token_id in vocab.items():
            if token.startswith(prefix) and token[prefix_len:].isdigit():
                bin_token_ids.add(token_id)
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"Could not extract bin tokens: {e}")
    return bin_token_ids


class NumericalAwareEmbedding(nn.Module):
    """Embedding with magnitude scaling for bin tokens (MNT)."""
    
    def __init__(self, tokenizer=None, vocab_size=None, 
                 original_embedding: nn.Embedding = None,
                 use_numerical_scaling: bool = True,
                 mnt_encoder=None):
        super().__init__()
        
        if original_embedding is not None:
            self.original_embedding = original_embedding
            self.hidden_size = original_embedding.embedding_dim
        else:
            self.hidden_size = 768
            if vocab_size is not None:
                self.original_embedding = nn.Embedding(vocab_size, self.hidden_size)
            else:
                raise ValueError("Must provide either original_embedding or vocab_size")
        
        self.tokenizer = tokenizer
        self.mnt_encoder = mnt_encoder
        self.use_numerical_scaling = use_numerical_scaling
        
        self._bin_token_ids: Set[int] = build_bin_token_set(tokenizer)
        self._bin_token_tensor: Optional[torch.Tensor] = None
        
        self.rms_norm = LlamaRMSNorm(self.hidden_size, eps=1e-6)
        
        if self.original_embedding is not None:
            dtype = self.original_embedding.weight.dtype
            device = self.original_embedding.weight.device
            self.rms_norm = self.rms_norm.to(dtype=dtype, device=device)
        
        # The backbone calls embeddings without forwarding input_scales.
        self.current_input_scales: Optional[torch.Tensor] = None
    
    
    def forward(self, input_ids: torch.Tensor, 
                input_scales: Optional[torch.Tensor] = None,
                is_mnt_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scale bin embeddings, then apply RMSNorm to all token embeddings."""
        if input_scales is None:
            input_scales = self.current_input_scales
            
        embeddings = self.original_embedding(input_ids)
        target_dtype = embeddings.dtype
        
        if (self.rms_norm.weight.dtype != embeddings.dtype or
                self.rms_norm.weight.device != embeddings.device):
            self.rms_norm = self.rms_norm.to(dtype=embeddings.dtype, device=embeddings.device)
        
        if is_mnt_token is None:
            is_bin_token = self._detect_bin_tokens(input_ids)
        else:
            is_bin_token = is_mnt_token
        
        # Only bin tokens receive multiplicative modulation.
        if is_bin_token.any():
            bin_mask = is_bin_token.unsqueeze(-1).expand_as(embeddings)
            bin_embeddings = embeddings[bin_mask].view(-1, embeddings.size(-1))
            
            bin_mnt_embeddings = bin_embeddings
            
            if self.use_numerical_scaling and input_scales is not None:
                input_scales = input_scales.to(device=is_bin_token.device)
                # Match the selected embeddings before multiplication.
                bin_scales = input_scales[is_bin_token].to(
                    device=bin_mnt_embeddings.device,
                    dtype=bin_mnt_embeddings.dtype,
                )
                bin_mnt_embeddings = bin_mnt_embeddings * bin_scales.unsqueeze(-1)
            
            processed_embeddings = embeddings.clone()
            processed_embeddings[bin_mask] = bin_mnt_embeddings.to(target_dtype).view(-1)
        else:
            processed_embeddings = embeddings
        
        processed_embeddings = self.rms_norm(processed_embeddings)

        return processed_embeddings.to(target_dtype)
    
    def _detect_bin_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Detect bin tokens using cached token IDs."""
        if not self._bin_token_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        
        if self._bin_token_tensor is None or self._bin_token_tensor.device != input_ids.device:
            self._bin_token_tensor = torch.tensor(
                list(self._bin_token_ids), device=input_ids.device, dtype=torch.long
            )
        
        return torch.isin(input_ids, self._bin_token_tensor)




class NumericalAwareModelWrapper(nn.Module):
    """Model wrapper that passes input_scales to embedding layer."""
    
    def __init__(self, original_model: nn.Module, embedding_wrapper: nn.Module):
        super().__init__()
        self._original_model = original_model
        self._embedding_wrapper = embedding_wrapper
    
    def __getattr__(self, name: str):
        if name.startswith('_'):
            return super().__getattr__(name)
        try:
            return getattr(self._original_model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, input_scales: Optional[torch.Tensor] = None, **kwargs):
        """Forward with numerical parameter handling."""
        self._embedding_wrapper.current_input_scales = input_scales
        result = self._original_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        self._embedding_wrapper.current_input_scales = None
        return result


def wrap_model_with_numerical_awareness(model, tokenizer, mnt_config=None, 
                                        use_numerical_scaling=True,
                                        mnt_encoder=None):
    """
    Enhance language model with numerical awareness capabilities
    Replaces embedding layers with numerical-aware versions
    
    Args:
        model: Original language model (SmolLM)
        tokenizer: Tokenizer (should include MNT tokens if using MNT)
        mnt_config: MNT configuration (optional)
        use_numerical_scaling: Whether to enable numerical scaling
        mnt_encoder: MNT encoder instance for token detection
        
    Returns:
        Enhanced model with numerical awareness
    """
    embedding_wrapped = False
    
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        original_embedding = model.model.embed_tokens
        model.model.embed_tokens = NumericalAwareEmbedding(
            tokenizer=tokenizer,
            original_embedding=original_embedding, 
            use_numerical_scaling=use_numerical_scaling,
            mnt_encoder=mnt_encoder
        )
        logger.info("Applied numerical-aware embeddings to model.embed_tokens")
        embedding_wrapped = True
    else:
        logger.warning("Could not find SmolLM embedding layer (model.model.embed_tokens). Model structure may not be supported.")
    

    
    if embedding_wrapped:
        numerical_wrapper = model.model.embed_tokens
        model = NumericalAwareModelWrapper(model, numerical_wrapper)
        logger.info("Wrapped model with numerical parameter handling")
    
    return model


class AnoLLMNumericalDataCollator:
    """Pad token features and numerical scales for AnoLLM batches."""
    
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None,
                 use_input_scales: bool = True, 
                 default_dtype: torch.dtype = torch.float32):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            pad_to_multiple_of: Padding alignment
            use_input_scales: Whether to add input_scales to batch
            default_dtype: Default dtype for input_scales tensor (default: float32)
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_input_scales = use_input_scales
        self.default_dtype = default_dtype
        
        self.bin_token_ids: Set[int] = build_bin_token_set(tokenizer)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Align optional input scales with padded tokens using neutral scale 1.0."""
        input_scales_data: List[Optional[List[float]]] = []
        has_scaling_data = False
        
        for feature in features:
            if 'input_scales' in feature:
                scales = feature.pop('input_scales')
                input_scales_data.append(scales)
                has_scaling_data = True
            else:
                input_scales_data.append(None)
        
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()
        
        if self.use_input_scales and (has_scaling_data or self.bin_token_ids):
            max_length = batch['input_ids'].shape[1]
            batch_size = batch['input_ids'].shape[0]
            
            # Missing scales and padding remain neutral.
            batch_input_scales = torch.ones((batch_size, max_length), dtype=self.default_dtype)
            
            for i in range(batch_size):
                input_ids_seq = batch['input_ids'][i]
                
                has_bin_tokens = any(token_id.item() in self.bin_token_ids for token_id in input_ids_seq)
                
                if has_bin_tokens and i < len(input_scales_data):
                    scales = input_scales_data[i]
                    if scales is not None:
                        seq_length = min(len(scales), max_length)
                        batch_input_scales[i, :seq_length] = torch.tensor(
                            scales[:seq_length], dtype=self.default_dtype
                        )
            
            batch['input_scales'] = batch_input_scales
        
        return batch


@dataclass 
class MNTConfig:
    """Configuration for MNT"""
    max_bins: int = 128  # Shared vocabulary size and per-feature bin cap.
    binning_strategy: str = "kmeans"
    bayesian_blocks_p0: float = 0.05
    start_token_id: int = None  # Set when the contiguous bin vocabulary is registered.
    mnt_token_prefix: str = "bin"
    

class MNTEncoder:
    """
    MNT encoder
    Converts numerical features to discrete tokens based on learned bins
    """
    
    def __init__(self, config: MNTConfig):
        self.config = config
        self.bin_edges = {}
        self.effective_bins = {}
        self.occupied_bins = {}
        self.binning_metadata = {}
        self.fitted = False
        self.feature_names = []

    def _transform_numeric_array(self, col_name: str, feature_values: Union[pd.Series, np.ndarray, List[Any]]) -> np.ndarray:
        """Transform a single feature array using the fitted bin edges."""
        if not self.fitted:
            raise ValueError("MNTEncoder must be fitted before transform")
        if col_name not in self.bin_edges:
            raise ValueError(f"Feature {col_name} is not fitted in MNTEncoder")

        numeric_values = pd.to_numeric(pd.Series(feature_values), errors="coerce").to_numpy(dtype=np.float64)
        bin_edges = self.bin_edges[col_name]

        return np.digitize(
            numeric_values,
            bins=np.r_[-np.inf, bin_edges[1:-1], np.inf]
        ).astype(np.int64) - 1

    def transform_feature(self, col_name: str, feature_values: Union[pd.Series, np.ndarray, List[Any]]) -> np.ndarray:
        """Transform a single feature into token ids using the shared MNT logic."""
        if self.config.start_token_id is None:
            raise ValueError("MNTConfig.start_token_id must be set before transforming values")
        return self._transform_numeric_array(col_name, feature_values) + self.config.start_token_id

    def transform_value(self, col_name: str, value: Any) -> int:
        """Transform a single scalar feature value into a token id."""
        return int(self.transform_feature(col_name, [value])[0])

    def _build_bin_edges_with_kbins(self, s: pd.Series, strategy: str) -> np.ndarray:
        """Build bin edges using KBinsDiscretizer for the requested strategy."""
        from sklearn.preprocessing import KBinsDiscretizer

        unique_values = s.drop_duplicates().sort_values().reset_index(drop=True)
        n_unique = len(unique_values)

        if n_unique == 0 or n_unique == 1:
            c = float(unique_values.iloc[0]) if not unique_values.empty else 0.0
            return np.array([c, c], dtype=np.float64)

        n_bins = min(self.config.max_bins, n_unique)
        values = s.to_numpy(dtype=np.float64).reshape(-1, 1)
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy=strategy,
        )
        discretizer.fit(values)
        return np.asarray(discretizer.bin_edges_[0], dtype=np.float64)

    def _build_bin_edges_with_ckmeans(self, s: pd.Series) -> np.ndarray:
        """Build globally optimal one-dimensional k-means bin edges."""
        try:
            from ckmeans import ckmeans
        except ImportError as exc:
            raise ImportError(
                "The 'ckmeans' package is required for "
                "mnt_binning_strategy='ckmeans_1d_dp'."
            ) from exc

        values = s.to_numpy(dtype=np.float64)
        unique_values = np.unique(values)
        if unique_values.size <= 1:
            c = float(unique_values[0]) if unique_values.size else 0.0
            return np.array([c, c], dtype=np.float64)

        n_bins = min(self.config.max_bins, unique_values.size)
        if n_bins == 1:
            return np.array([unique_values[0], unique_values[-1]], dtype=np.float64)

        clusters = ckmeans(values, n_bins)
        if len(clusters) != n_bins:
            raise RuntimeError(
                f"Ckmeans returned {len(clusters)} clusters; expected {n_bins}."
            )

        ordered_clusters = sorted(
            (np.asarray(cluster, dtype=np.float64) for cluster in clusters),
            key=lambda cluster: float(np.min(cluster)),
        )
        internal_edges = []
        for left_cluster, right_cluster in zip(ordered_clusters[:-1], ordered_clusters[1:]):
            left_max = float(np.max(left_cluster))
            right_min = float(np.min(right_cluster))
            internal_edges.append(left_max + (right_min - left_max) / 2.0)

        edges = np.asarray(
            [unique_values[0], *internal_edges, unique_values[-1]],
            dtype=np.float64,
        )
        if not np.all(np.diff(edges) > 0):
            raise RuntimeError("Ckmeans produced non-increasing bin edges.")
        return edges

    def _build_bin_edges_with_bayesian_blocks(self, s: pd.Series) -> tuple[np.ndarray, float]:
        """Build adaptive Bayesian Blocks edges under the shared token cap."""
        try:
            from astropy.stats import bayesian_blocks
        except ImportError as exc:
            raise ImportError(
                "The 'astropy' package is required for "
                "mnt_binning_strategy='bayesian_blocks'."
            ) from exc

        values = s.to_numpy(dtype=np.float64)
        unique_values = np.unique(values)
        if unique_values.size <= 1:
            c = float(unique_values[0]) if unique_values.size else 0.0
            return np.array([c, c], dtype=np.float64), self.config.bayesian_blocks_p0

        requested_p0 = float(getattr(self.config, "bayesian_blocks_p0", 0.05))
        if not 0.0 < requested_p0 < 1.0:
            raise ValueError("bayesian_blocks_p0 must be between 0 and 1.")

        # Tighten the false-alarm prior if adaptive bins exceed the token budget.
        effective_p0 = requested_p0
        for _ in range(32):
            edges = np.asarray(
                bayesian_blocks(values, fitness="events", p0=effective_p0),
                dtype=np.float64,
            )
            edges = np.unique(edges[np.isfinite(edges)])
            if edges.size >= 2 and edges.size - 1 <= self.config.max_bins:
                return edges, effective_p0
            effective_p0 *= 0.5

        raise RuntimeError(
            "Bayesian Blocks could not satisfy mnt_max_bins after tightening p0."
        )
        
    def fit(self, X: pd.DataFrame):
        """
        Fit the MNT encoder on training data using configurable discretization.
        
        Supported strategies:
        - kmeans: cluster values and place edges between cluster centers
        - quantile: place edges at quantile cut points
        - uniform: split the observed range into equal-width intervals
        - ckmeans_1d_dp: globally optimal one-dimensional k-means
        - bayesian_blocks: adaptive density segmentation
        
        Args:
            X: DataFrame with numerical features
        """
        strategy = getattr(self.config, "binning_strategy", "kmeans")
        self.config.binning_strategy = strategy

        valid_strategies = {
            "kmeans",
            "quantile",
            "uniform",
            "ckmeans_1d_dp",
            "bayesian_blocks",
        }
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid MNT binning strategy: {strategy}. "
                f"Choose from {sorted(valid_strategies)}."
            )

        self.feature_names = list(X.columns)
        self.bin_edges = {}
        self.effective_bins = {}
        self.occupied_bins = {}
        self.binning_metadata = {}

        progress_desc = f"Fitting MNT bins ({strategy})"
        for col_name in tqdm(self.feature_names, desc=progress_desc):
            s = pd.to_numeric(X[col_name], errors="coerce")
            s = s[np.isfinite(s)]

            if strategy in {"kmeans", "quantile", "uniform"}:
                edges = self._build_bin_edges_with_kbins(s, strategy)
            elif strategy == "ckmeans_1d_dp":
                edges = self._build_bin_edges_with_ckmeans(s)
            else:
                edges, effective_p0 = self._build_bin_edges_with_bayesian_blocks(s)
                self.binning_metadata[col_name] = {
                    "requested_p0": float(getattr(self.config, "bayesian_blocks_p0", 0.05)),
                    "effective_p0": effective_p0,
                }
            
            self.bin_edges[col_name] = edges
            self.effective_bins[col_name] = max(1, len(edges) - 1)
            train_bin_ids = np.digitize(
                s.to_numpy(dtype=np.float64),
                bins=np.r_[-np.inf, edges[1:-1], np.inf],
            ).astype(np.int64) - 1
            self.occupied_bins[col_name] = len(np.unique(train_bin_ids))
            
        self.fitted = True
        
        print("\n" + "="*80)
        print(f"MNT Binning Summary ({strategy})")
        print("="*80)
        print(f"{'Feature Name':<52} {'Unique':<10} {'Bins':<10} {'Occupied':<10}")
        print("-"*80)
        for col_name in self.feature_names:
            s = pd.to_numeric(X[col_name], errors="coerce")
            s = s[np.isfinite(s)]
            n_unique = len(s.unique())
            n_bins = self.effective_bins[col_name]
            n_occupied = self.occupied_bins[col_name]
            display_name = col_name[:49] + "..." if len(col_name) > 52 else col_name
            print(f"{display_name:<52} {n_unique:<10} {n_bins:<10} {n_occupied:<10}")
        print("-"*80)
        print(f"Total features: {len(self.feature_names)}, Max bins config: {self.config.max_bins}")
        print("="*80)
        
    def transform(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform numerical features to IDs in the shared MNT bin vocabulary
        
        Args:
            X: DataFrame with numerical features
            
        Returns:
            Dictionary mapping feature names to token IDs
        """
        if not self.fitted:
            raise ValueError("MNTEncoder must be fitted before transform")
            
        mnt_tokens = {}
        
        for feature_idx, col_name in enumerate(self.feature_names):
            if col_name not in X.columns:
                raise ValueError(f"Feature {col_name} not found in input data")
            mnt_tokens[col_name] = self.transform_feature(col_name, X[col_name].values)
                
        return mnt_tokens
    
    def fit_transform(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def get_vocab_size(self) -> int:
        """Get the number of MNT tokens needed (shared across all features)"""
        return self.config.max_bins
    
    def save(self, filepath: str):
        """Save the fitted encoder"""
        if not self.fitted:
            raise ValueError("Cannot save unfitted encoder")
            
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'bin_edges': self.bin_edges,
                'effective_bins': self.effective_bins,
                'occupied_bins': self.occupied_bins,
                'binning_metadata': self.binning_metadata,
                'feature_names': self.feature_names,
                'fitted': self.fitted
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted encoder"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        encoder = cls(data['config'])
        if not hasattr(encoder.config, 'binning_strategy'):
            encoder.config.binning_strategy = 'kmeans'
        encoder.bin_edges = data['bin_edges']
        encoder.effective_bins = data.get('effective_bins', {})
        encoder.occupied_bins = data.get('occupied_bins', {})
        encoder.binning_metadata = data.get('binning_metadata', {})
        encoder.feature_names = data['feature_names']
        encoder.fitted = data['fitted']
        
        return encoder


def add_mnt_tokens_to_tokenizer(tokenizer, config: MNTConfig):
    """Add the shared bin vocabulary and record its contiguous token range."""
    mnt_tokens = [f"{config.mnt_token_prefix}{i}" for i in range(config.max_bins)]
    vocab = tokenizer.get_vocab()
    new_tokens = [token for token in mnt_tokens if token not in vocab]

    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {len(new_tokens)} MNT tokens to tokenizer ({mnt_tokens[0]} to {mnt_tokens[-1]})")
        logger.info(f"All numerical features will share these {config.max_bins} tokens")

    token_ids = tokenizer.convert_tokens_to_ids(mnt_tokens)
    start_token_id = token_ids[0]
    if token_ids != list(range(start_token_id, start_token_id + config.max_bins)):
        raise ValueError("MNT bin token IDs must be contiguous in the tokenizer vocabulary")
    config.start_token_id = start_token_id

    return tokenizer, new_tokens
