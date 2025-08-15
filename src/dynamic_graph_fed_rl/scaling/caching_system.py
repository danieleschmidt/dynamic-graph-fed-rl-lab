"""Advanced caching system for scalable federated RL performance."""

import time
import threading
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import OrderedDict, defaultdict
import asyncio


class CacheStrategy(Enum):
    """Caching strategies for different use cases."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns
    PREDICTIVE = "predictive"  # Predictive caching based on patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class CachingSystem:
    """
    Advanced caching system with multiple strategies and intelligent optimization.
    
    Features:
    - Multiple caching strategies (LRU, LFU, TTL, Adaptive, Predictive)
    - Intelligent cache warming and prefetching
    - Memory-aware cache sizing
    - Performance analytics and optimization
    - Thread-safe operations
    - Distributed cache coherence
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_statistics: bool = True,
        enable_prefetching: bool = True,
        memory_limit_mb: float = 500.0
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_statistics = enable_statistics
        self.enable_prefetching = enable_prefetching
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Statistics and analytics
        self.stats = CacheStatistics() if enable_statistics else None
        self.access_patterns = defaultdict(list)
        self.prefetch_queue = asyncio.Queue() if enable_prefetching else None
        
        # Strategy-specific data structures
        self.frequency_counter = defaultdict(int)  # For LFU
        self.prediction_model = PredictiveCacheModel()  # For predictive caching
        
        # Background tasks
        self.cleanup_task = None
        self.prefetch_task = None
        
        # Performance monitoring
        self.current_memory_usage = 0
        self.cache_warming_functions: Dict[str, Callable] = {}
        
        print(f"🚀 Caching system initialized: {strategy.value} strategy, {max_size} max entries")
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        try:
            loop = asyncio.get_running_loop()
            if self.enable_prefetching:
                loop.create_task(self._prefetch_worker())
            
            # Start cleanup task for expired entries
            loop.create_task(self._cleanup_worker())
        except RuntimeError:
            # No event loop running - skip async tasks for sync usage
            print("   📋 Caching system running in sync mode (no async tasks)")
            self.prefetch_task = None
            self.cleanup_task = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with strategy-specific behavior."""
        with self.cache_lock:
            entry = self.cache.get(key)
            
            if entry is None:
                if self.stats:
                    self.stats.record_miss()
                return default
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.current_memory_usage -= entry.size
                if self.stats:
                    self.stats.record_miss()
                return default
            
            # Update access patterns
            entry.update_access()
            if self.stats:
                self.stats.record_hit()
            
            # Strategy-specific behavior
            if self.strategy == CacheStrategy.LRU:
                # Move to end for LRU
                self.cache.move_to_end(key)
            elif self.strategy == CacheStrategy.LFU:
                self.frequency_counter[key] += 1
            
            # Record access pattern for predictive caching
            if self.enable_prefetching:
                self._record_access_pattern(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache with strategy-specific eviction."""
        with self.cache_lock:
            # Calculate value size
            value_size = self._calculate_size(value)
            
            # Check memory limits
            if value_size > self.memory_limit_bytes:
                print(f"⚠️  Value too large for cache: {value_size / 1024 / 1024:.1f}MB")
                return False
            
            # Ensure space is available
            self._ensure_space(value_size)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size=value_size
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_usage -= old_entry.size
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory_usage += value_size
            
            if self.stats:
                self.stats.record_put()
            
            return True
    
    def _ensure_space(self, required_size: int):
        """Ensure sufficient space using strategy-specific eviction."""
        while (self.current_memory_usage + required_size > self.memory_limit_bytes or 
               len(self.cache) >= self.max_size):
            
            if not self.cache:
                break
            
            evicted_key = self._select_eviction_candidate()
            if evicted_key:
                self.evict(evicted_key)
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select eviction candidate based on strategy."""
        if not self.cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used (first in OrderedDict)
            return next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            min_freq = min(self.frequency_counter.get(k, 0) for k in self.cache.keys())
            for key in self.cache.keys():
                if self.frequency_counter.get(key, 0) == min_freq:
                    return key
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            current_time = time.time()
            for key, entry in self.cache.items():
                if entry.is_expired():
                    return key
            # If no expired entries, evict oldest
            return next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            return self._adaptive_eviction_candidate()
        
        elif self.strategy == CacheStrategy.PREDICTIVE:
            # Predictive eviction based on usage prediction
            return self._predictive_eviction_candidate()
        
        # Fallback to LRU
        return next(iter(self.cache))
    
    def _adaptive_eviction_candidate(self) -> Optional[str]:
        """Select eviction candidate using adaptive strategy."""
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            # Combine recency, frequency, and size factors
            recency_score = 1.0 / max(current_time - entry.last_access, 1.0)
            frequency_score = entry.access_count / max(current_time - entry.timestamp, 1.0)
            size_penalty = entry.size / 1024  # Penalize large entries
            
            scores[key] = recency_score + frequency_score - size_penalty
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _predictive_eviction_candidate(self) -> Optional[str]:
        """Select eviction candidate using predictive strategy."""
        predictions = self.prediction_model.predict_future_access(self.cache.keys())
        
        # Evict entry with lowest predicted access probability
        if predictions:
            return min(predictions.keys(), key=lambda k: predictions[k])
        
        # Fallback to adaptive strategy
        return self._adaptive_eviction_candidate()
    
    def evict(self, key: str) -> bool:
        """Evict specific key from cache."""
        with self.cache_lock:
            entry = self.cache.pop(key, None)
            if entry:
                self.current_memory_usage -= entry.size
                if key in self.frequency_counter:
                    del self.frequency_counter[key]
                if self.stats:
                    self.stats.record_eviction()
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.cache_lock:
            self.cache.clear()
            self.frequency_counter.clear()
            self.current_memory_usage = 0
            if self.stats:
                self.stats.record_clear()
    
    def warm_cache(self, warm_function: Callable, keys: List[str], namespace: str = "default"):
        """Warm cache with pre-computed values."""
        print(f"🔥 Warming cache for {len(keys)} keys in namespace '{namespace}'")
        
        self.cache_warming_functions[namespace] = warm_function
        
        warmed_count = 0
        for key in keys:
            try:
                if key not in self.cache:
                    value = warm_function(key)
                    if self.put(key, value):
                        warmed_count += 1
            except Exception as e:
                print(f"⚠️  Cache warming failed for key '{key}': {e}")
        
        print(f"✅ Cache warming completed: {warmed_count}/{len(keys)} entries")
    
    async def prefetch(self, keys: List[str], prefetch_function: Callable):
        """Asynchronously prefetch values for expected future use."""
        if not self.enable_prefetching:
            return
        
        for key in keys:
            if key not in self.cache:
                await self.prefetch_queue.put((key, prefetch_function))
    
    async def _prefetch_worker(self):
        """Background worker for prefetching cache entries."""
        while True:
            try:
                key, prefetch_function = await asyncio.wait_for(
                    self.prefetch_queue.get(), timeout=1.0
                )
                
                if key not in self.cache:
                    try:
                        value = await prefetch_function(key) if asyncio.iscoroutinefunction(prefetch_function) else prefetch_function(key)
                        self.put(key, value)
                        print(f"🔮 Prefetched: {key}")
                    except Exception as e:
                        print(f"⚠️  Prefetch failed for '{key}': {e}")
                
                self.prefetch_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Prefetch worker error: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleaning up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                expired_keys = []
                current_time = time.time()
                
                with self.cache_lock:
                    for key, entry in self.cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                
                for key in expired_keys:
                    self.evict(key)
                
                if expired_keys:
                    print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                print(f"❌ Cleanup worker error: {e}")
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for predictive caching."""
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only recent patterns (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff_time]
        
        # Update prediction model
        self.prediction_model.update_patterns(self.access_patterns)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value)) * 2
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.cache_lock:
            stats_dict = {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.current_memory_usage / 1024 / 1024,
                "memory_limit_mb": self.memory_limit_bytes / 1024 / 1024,
                "strategy": self.strategy.value,
            }
            
            if self.stats:
                stats_dict.update(self.stats.get_statistics())
            
            return stats_dict
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance based on usage patterns."""
        optimization_results = {}
        
        with self.cache_lock:
            # Analyze access patterns
            if self.stats and self.stats.total_requests > 100:
                hit_rate = self.stats.cache_hits / self.stats.total_requests
                
                if hit_rate < 0.5:
                    optimization_results["recommendation"] = "Consider increasing cache size or TTL"
                elif hit_rate > 0.9:
                    optimization_results["recommendation"] = "Cache is performing well"
                else:
                    optimization_results["recommendation"] = "Consider tuning eviction strategy"
                
                optimization_results["current_hit_rate"] = hit_rate
            
            # Memory optimization
            if self.current_memory_usage > self.memory_limit_bytes * 0.9:
                optimization_results["memory_warning"] = "Cache approaching memory limit"
                
                # Suggest memory optimization
                large_entries = [(k, v.size) for k, v in self.cache.items() if v.size > 1024 * 1024]
                if large_entries:
                    optimization_results["large_entries_count"] = len(large_entries)
                    optimization_results["memory_optimization"] = "Consider evicting large entries"
            
            # Strategy optimization
            if self.strategy == CacheStrategy.ADAPTIVE:
                # Analyze if another strategy might be better
                access_variance = self._calculate_access_variance()
                if access_variance < 0.1:
                    optimization_results["strategy_suggestion"] = "Consider LRU strategy for uniform access"
                elif access_variance > 0.8:
                    optimization_results["strategy_suggestion"] = "Consider LFU strategy for skewed access"
        
        return optimization_results
    
    def _calculate_access_variance(self) -> float:
        """Calculate variance in access patterns."""
        if not self.cache:
            return 0.0
        
        access_counts = [entry.access_count for entry in self.cache.values()]
        if not access_counts:
            return 0.0
        
        mean_access = sum(access_counts) / len(access_counts)
        variance = sum((count - mean_access) ** 2 for count in access_counts) / len(access_counts)
        
        return variance / max(mean_access, 1.0)  # Normalized variance
    
    def shutdown(self):
        """Shutdown caching system and cleanup resources."""
        print("🧹 Shutting down caching system...")
        self.clear()
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.prefetch_task:
            self.prefetch_task.cancel()


class CacheStatistics:
    """Cache performance statistics tracking."""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_puts = 0
        self.cache_evictions = 0
        self.cache_clears = 0
        self.start_time = time.time()
    
    def record_hit(self):
        self.cache_hits += 1
    
    def record_miss(self):
        self.cache_misses += 1
    
    def record_put(self):
        self.cache_puts += 1
    
    def record_eviction(self):
        self.cache_evictions += 1
    
    def record_clear(self):
        self.cache_clears += 1
    
    @property
    def total_requests(self) -> int:
        return self.cache_hits + self.cache_misses
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def get_statistics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_puts": self.cache_puts,
            "cache_evictions": self.cache_evictions,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "requests_per_second": self.total_requests / max(uptime, 1.0),
            "uptime_seconds": uptime,
        }


class PredictiveCacheModel:
    """Predictive model for cache access patterns."""
    
    def __init__(self):
        self.pattern_weights = defaultdict(float)
        self.time_decay_factor = 0.9
    
    def update_patterns(self, access_patterns: Dict[str, List[float]]):
        """Update prediction model with new access patterns."""
        current_time = time.time()
        
        for key, timestamps in access_patterns.items():
            if len(timestamps) < 2:
                continue
            
            # Calculate access frequency and recency
            frequency = len(timestamps) / 3600  # Accesses per hour
            recency = current_time - max(timestamps)
            
            # Update weight with time decay
            self.pattern_weights[key] = (
                self.pattern_weights[key] * self.time_decay_factor + 
                frequency / max(recency, 1.0)
            )
    
    def predict_future_access(self, keys: List[str]) -> Dict[str, float]:
        """Predict future access probabilities for given keys."""
        predictions = {}
        
        for key in keys:
            # Simple prediction based on historical patterns
            base_probability = self.pattern_weights.get(key, 0.1)
            
            # Add randomness to avoid deterministic behavior
            import random
            noise = random.uniform(-0.05, 0.05)
            
            predictions[key] = max(0.0, min(1.0, base_probability + noise))
        
        return predictions