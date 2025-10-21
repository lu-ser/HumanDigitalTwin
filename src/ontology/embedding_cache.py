"""
Persistent cache for embeddings to avoid rate limits.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


class EmbeddingCache:
    _instance_counter = 0

    def __init__(self, cache_dir: str = "data/cache", provider: str = "cohere", shared_cache: bool = False):
        EmbeddingCache._instance_counter += 1
        self.instance_id = EmbeddingCache._instance_counter

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        # Use shared cache or separate cache per provider
        if shared_cache:
            self.cache_file = self.cache_dir / "embeddings_cache_shared.pkl"
        else:
            self.cache_file = self.cache_dir / f"embeddings_cache_{provider}.pkl"

        # Setup logging
        self.log_file = self.cache_dir / f"cache_debug_{provider}.log"

        self.cache: Dict[str, List[float]] = {}
        self._log(f"🆔 [CACHE INSTANCE #{self.instance_id}] Created for provider={provider}, file={self.cache_file}")
        self.load_cache()

    def _log(self, message: str):
        """Log to both console and file."""
        try:
            print(message, flush=True)
        except UnicodeEncodeError:
            # Fallback for Windows console encoding issues
            print(message.encode('ascii', 'replace').decode('ascii'), flush=True)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass

    def load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"[Cache] Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"[Warning] Could not load cache: {e}")
                self.cache = {}

    def save_cache(self):
        """Save cache to disk, verify, and reload into memory."""
        try:
            cache_size_before = len(self.cache)

            # Save to disk with explicit flush
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Force OS to write to disk

            self._log(f"💾 [CACHE #{self.instance_id} SAVE] Saved {cache_size_before} embeddings to {self.cache_file}")

            # FORCE reload from disk to ensure consistency
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)

            self._log(f"✅ [CACHE #{self.instance_id} RELOAD] Reloaded {len(self.cache)} embeddings from disk into memory")

            if len(self.cache) != cache_size_before:
                print(f"⚠️ [CACHE VERIFY] Size mismatch! Saved {cache_size_before} but reloaded {len(self.cache)}")
        except Exception as e:
            print(f"⚠️ [CACHE SAVE ERROR] Could not save cache: {e}")

    def _make_key(self, text: str, input_type: str = "search_document") -> str:
        """Create a unique key for caching."""
        content = f"{input_type}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, input_type: str = "search_document") -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._make_key(text, input_type)
        result = self.cache.get(key)
        if result is not None:
            self._log(f"✅ [CACHE #{self.instance_id} GET] Found: {text[:30]}...")
        else:
            self._log(f"❌ [CACHE #{self.instance_id} GET] NOT found: {text[:30]}... (key: {key[:8]}, cache_size: {len(self.cache)})")
        return result

    def set(self, text: str, embedding: List[float], input_type: str = "search_document"):
        """Store embedding in cache."""
        key = self._make_key(text, input_type)
        self.cache[key] = embedding
        self._log(f"💾 [CACHE #{self.instance_id} SET] Stored: {text[:30]}... (key: {key[:8]})")

    def get_batch(self, texts: List[str], input_type: str = "search_document") -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings from cache."""
        results = {}
        for text in texts:
            results[text] = self.get(text, input_type)
        return results

    def set_batch(self, texts: List[str], embeddings: List[List[float]], input_type: str = "search_document"):
        """Store multiple embeddings in cache."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, input_type)

    def clear(self):
        """Clear all cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
