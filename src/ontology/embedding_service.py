"""
Embedding service with support for multiple providers (Cohere, Mistral, Gemma).
"""

import os
import time
from typing import List, Dict, Optional
from enum import Enum
import cohere
from mistralai import Mistral
from dotenv import load_dotenv
from .embedding_cache import EmbeddingCache

# Try importing transformers for Gemma
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

# Load environment variables
load_dotenv(override=True)


class EmbeddingProvider(Enum):
    COHERE = "cohere"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    QWEN = "qwen"


class EmbeddingService:
    def __init__(
        self,
        provider: str = "cohere",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "data/cache",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        shared_cache: bool = True
    ):
        """
        Initialize embedding service.

        Args:
            provider: "cohere", "mistral", "gemma", or "qwen"
            api_key: API key for the provider (or reads from env)
            use_cache: Whether to use persistent caching
            cache_dir: Directory for cache storage (default: "data/cache")
            model_path: Local path for Gemma/Qwen model (optional, defaults to HF hub)
            device: Device to use for Gemma/Qwen ("cuda", "cpu", or None for auto-detect)
            shared_cache: Use shared cache across providers (default True)
        """
        self.provider = EmbeddingProvider(provider)
        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir=cache_dir, provider=provider, shared_cache=False) if use_cache else None

        # Initialize the appropriate client
        if self.provider == EmbeddingProvider.COHERE:
            self.api_key = api_key or os.getenv('COHERE_API_KEY')
            if not self.api_key:
                raise ValueError("Cohere API key not found. Set COHERE_API_KEY in .env")
            self.client = cohere.Client(self.api_key)
            self.model = "embed-english-v3.0"
            self.embedding_dim = 1024

        elif self.provider == EmbeddingProvider.MISTRAL:
            self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
            if not self.api_key:
                raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY in .env")
            self.client = Mistral(api_key=self.api_key)
            self.model = "mistral-embed"
            self.embedding_dim = 1024

        elif self.provider == EmbeddingProvider.GEMMA:
            if not GEMMA_AVAILABLE:
                raise ImportError(
                    "Gemma requires transformers and torch. Install with: "
                    "pip install transformers torch accelerate"
                )

            # Get HuggingFace token for gated models
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError(
                    "HuggingFace token not found. Set HF_TOKEN in .env\n"
                    "Get your token from: https://huggingface.co/settings/tokens\n"
                    "You also need to request access to the Gemma model at:\n"
                    "https://huggingface.co/google/embeddinggemma-300m"
                )

            # Use local path or HuggingFace model
            self.model = model_path or "google/embeddinggemma-300m"

            # Load tokenizer and model with authentication
            print(f"Loading Gemma model from {self.model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                token=hf_token
            )
            # Load model with compatibility settings for PyTorch < 2.6
            import transformers
            self.client = AutoModel.from_pretrained(
                self.model,
                token=hf_token,
                attn_implementation="eager",  # Use eager attention instead of SDPA
                torch_dtype=torch.float32,  # Ensure compatibility
                trust_remote_code=False
            )

            # Disable advanced masking features that require torch>=2.6
            if hasattr(self.client.config, 'use_mask_functions'):
                self.client.config.use_mask_functions = False

            # Set device (use provided device or auto-detect)
            if device is not None:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Using device: {self.device}")
            if self.device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")

            self.client = self.client.to(self.device)
            self.client.eval()

            self.embedding_dim = 768  # Gemma 300M produces 768-dim embeddings
            print(f"✅ Gemma model loaded on {self.device}")

        elif self.provider == EmbeddingProvider.QWEN:
            if not GEMMA_AVAILABLE:  # Reuse the same check
                raise ImportError(
                    "Qwen requires transformers and torch. Install with: "
                    "pip install transformers torch accelerate"
                )

            # Get HuggingFace token
            hf_token = os.getenv('HF_TOKEN')

            # Use local path or HuggingFace model
            self.model = model_path or "Qwen/Qwen3-Embedding-8B"

            # Load tokenizer and model
            print(f"Loading Qwen model from {self.model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                token=hf_token,
                trust_remote_code=True
            )
            self.client = AutoModel.from_pretrained(
                self.model,
                token=hf_token,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

            # Set device (use provided device or auto-detect)
            if device is not None:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Using device: {self.device}")
            if self.device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")

            self.client = self.client.to(self.device)
            self.client.eval()

            self.embedding_dim = 8192  # Qwen3-Embedding-8B produces 8192-dim embeddings
            print(f"✅ Qwen model loaded on {self.device}")

    def embed_texts(
        self,
        texts: List[str],
        input_type: str = "search_document",
        rate_limit_delay: float = 0.0
    ) -> List[List[float]]:
        """
        Generate embeddings for texts with caching and rate limiting.

        Args:
            texts: List of texts to embed
            input_type: "search_document" or "search_query" (Cohere only)
            rate_limit_delay: Delay in seconds between API calls

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache first
        if self.use_cache:
            cached_results = self.cache.get_batch(texts, input_type)
            uncached_texts = [t for t in texts if cached_results[t] is None]

            print(f"🔍 [CACHE] Total texts: {len(texts)}, Cached: {len(texts) - len(uncached_texts)}, Need to compute: {len(uncached_texts)}")

            # All cached - return immediately
            if not uncached_texts:
                print(f"✅ [CACHE] All embeddings found in cache!")
                result = [cached_results[t] for t in texts]
                # Ensure no None values
                return [emb if emb is not None else [0.0] * self.embedding_dim for emb in result]

            # Fetch uncached embeddings
            if uncached_texts:
                print(f"⚙️ [COMPUTE] Computing {len(uncached_texts)} new embeddings with {self.provider.value}...")
                try:
                    if rate_limit_delay > 0:
                        time.sleep(rate_limit_delay)

                    new_embeddings = self._fetch_embeddings(uncached_texts, input_type)

                    # Cache new embeddings ONLY for search_document (not queries)
                    if input_type == "search_document":
                        print(f"💾 [CACHE] Saving {len(uncached_texts)} new embeddings...")
                        self.cache.set_batch(uncached_texts, new_embeddings, input_type)
                        self.cache.save_cache()
                    else:
                        print(f"⚠️ [CACHE] Skipping save for input_type={input_type} (queries not cached)")

                    # Update results
                    for text, emb in zip(uncached_texts, new_embeddings):
                        cached_results[text] = emb

                except Exception as e:
                    print(f"⚠️ API Error: {e}")
                    print("Using cached results only.")
                    # Fill missing with zero vectors
                    for text in uncached_texts:
                        if cached_results.get(text) is None:
                            cached_results[text] = [0.0] * self.embedding_dim

            # Return all results, ensuring no None
            result = []
            for t in texts:
                emb = cached_results.get(t)
                if emb is not None:
                    result.append(emb)
                else:
                    result.append([0.0] * self.embedding_dim)
            return result

        # No cache - direct API call
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        return self._fetch_embeddings(texts, input_type)

    def _fetch_embeddings(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Fetch embeddings from the API or local model."""
        if self.provider == EmbeddingProvider.COHERE:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=input_type
            )
            return response.embeddings

        elif self.provider == EmbeddingProvider.MISTRAL:
            # Mistral API call
            response = self.client.embeddings.create(
                model=self.model,
                inputs=texts
            )
            return [item.embedding for item in response.data]

        elif self.provider == EmbeddingProvider.GEMMA:
            # Gemma local inference with batching
            embeddings = []
            batch_size = 64  # Process 64 texts at once on GPU

            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    print(f"⚙️ [GEMMA] Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} texts)")

                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    # Get embeddings
                    outputs = self.client(**inputs)

                    # Use mean pooling of last hidden state
                    # For batched inputs, mean over sequence dimension (dim=1)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                    # Convert to list
                    for emb in batch_embeddings:
                        embeddings.append(emb.cpu().numpy().tolist())

            return embeddings

        elif self.provider == EmbeddingProvider.QWEN:
            # Qwen local inference with batching
            embeddings = []
            batch_size = 32  # Smaller batch for larger model

            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    print(f"⚙️ [QWEN] Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} texts)")

                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    # Get embeddings
                    outputs = self.client(**inputs)

                    # Use mean pooling of last hidden state
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                    # Convert to list
                    for emb in batch_embeddings:
                        embeddings.append(emb.cpu().numpy().tolist())

            return embeddings

    def embed_text(
        self,
        text: str,
        input_type: str = "search_document",
        rate_limit_delay: float = 0.0
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            input_type: Type of input
            rate_limit_delay: Delay before API call

        Returns:
            Embedding vector
        """
        # For search queries (user input), don't save to cache
        if input_type == "search_query":
            # Check cache first
            if self.use_cache:
                cached = self.cache.get(text, input_type)
                if cached is not None:
                    return cached

            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)
            result = self._fetch_embeddings([text], input_type)
            return result[0] if result else [0.0] * self.embedding_dim

        # For documents, use normal caching flow (embed_texts handles caching)
        embeddings = self.embed_texts([text], input_type, rate_limit_delay)
        return embeddings[0] if embeddings else [0.0] * self.embedding_dim

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        # Robust handling of invalid inputs
        try:
            if not vec1 or not vec2:
                return 0.0

            # Filter out None values
            vec1_clean = [v if v is not None else 0.0 for v in vec1]
            vec2_clean = [v if v is not None else 0.0 for v in vec2]

            vec1_np = np.array(vec1_clean, dtype=np.float64)
            vec2_np = np.array(vec2_clean, dtype=np.float64)

            # Check for dimension mismatch
            if len(vec1_np) != len(vec2_np):
                return 0.0

            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            print(f"⚠️ Cosine similarity error: {e}")
            return 0.0
