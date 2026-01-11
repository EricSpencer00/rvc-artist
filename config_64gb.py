#!/usr/bin/env python3
"""
64GB RAM Machine Configuration for Multi-Stem Music Generation
===============================================================

This module provides optimized configurations for machines with 64GB+ RAM.
It enables parallel stem generation and model caching for maximum performance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config64GB:
    """Optimized config for 64GB RAM machines."""
    
    # Model settings
    model_size: str = "large"
    use_stereo: bool = True
    cache_all_models: bool = True  # Keep melody model in memory
    
    # Processing
    enable_processing: bool = True  # EQ, compression, reverb
    enable_mastering: bool = True   # LUFS normalization, limiting
    
    # Parallel generation (64GB+ RAM advantage)
    enable_parallel_generation: bool = True
    max_parallel_stems: int = 2    # Safely generate 2 stems at once
    
    # Generation parameters
    default_duration: int = 30
    default_temperature: float = 1.0
    default_guidance_scale: float = 3.5
    default_top_k: int = 250
    
    # Output
    save_individual_stems: bool = True
    output_dir: str = "./output/generated"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy passing to MultiStemGenerator."""
        return {
            'model_size': self.model_size,
            'use_stereo': self.use_stereo,
            'enable_processing': self.enable_processing,
            'enable_mastering': self.enable_mastering,
            'enable_parallel_generation': self.enable_parallel_generation,
            'max_parallel_stems': self.max_parallel_stems,
            'cache_all_models': self.cache_all_models
        }


# Preset configurations for different machine specs

CONFIG_64GB_OPTIMAL = Config64GB(
    model_size="large",
    max_parallel_stems=2,
    cache_all_models=True,
    enable_parallel_generation=True
)

CONFIG_64GB_BALANCED = Config64GB(
    model_size="medium",
    max_parallel_stems=3,
    cache_all_models=True,
    enable_parallel_generation=True
)

CONFIG_64GB_AGGRESSIVE = Config64GB(
    model_size="large",
    max_parallel_stems=4,
    cache_all_models=True,
    enable_parallel_generation=True
)

CONFIG_32GB = Config64GB(
    model_size="medium",
    max_parallel_stems=2,
    cache_all_models=False,  # Lazy load melody model
    enable_parallel_generation=True
)

CONFIG_16GB = Config64GB(
    model_size="small",
    max_parallel_stems=1,
    cache_all_models=False,
    enable_parallel_generation=False  # Sequential generation only
)


def get_config(machine_spec: str = "64gb") -> Config64GB:
    """Get configuration based on machine specs.
    
    Args:
        machine_spec: One of '64gb_optimal', '64gb_balanced', '64gb_aggressive',
                     '32gb', '16gb'
    
    Returns:
        Configured Config64GB instance
    """
    configs = {
        '64gb_optimal': CONFIG_64GB_OPTIMAL,
        '64gb_balanced': CONFIG_64GB_BALANCED,
        '64gb_aggressive': CONFIG_64GB_AGGRESSIVE,
        '32gb': CONFIG_32GB,
        '16gb': CONFIG_16GB,
    }
    
    return configs.get(machine_spec.lower(), CONFIG_64GB_OPTIMAL)


if __name__ == "__main__":
    from src.services.multi_stem_generator import MultiStemGenerator
    
    # Get optimal config for your 64GB machine
    config = get_config('64gb_optimal')
    
    print("64GB RAM Configuration")
    print("=" * 60)
    print(f"Model size: {config.model_size}")
    print(f"Parallel stems: {config.max_parallel_stems} at a time")
    print(f"Model caching: {config.cache_all_models}")
    print(f"Processing: {config.enable_processing}")
    print(f"Mastering: {config.enable_mastering}")
    print("=" * 60)
    
    # Initialize generator with 64GB settings
    generator = MultiStemGenerator(**config.to_dict())
    
    print("\n✅ Generator initialized with 64GB optimizations!")
    print(f"Ready to generate music with parallel stem generation.")
