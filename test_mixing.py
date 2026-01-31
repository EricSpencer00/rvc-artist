#!/usr/bin/env python3
"""Test mixing to find the audio amplitude issue."""

from src.services.multi_stem_generator import MultiStemGenerator
import numpy as np

gen = MultiStemGenerator(model_size="medium")

print("\nGenerating individual stems...")
stems = {}

for stem_type in ['drums', 'bass']:
    audio = gen.generate_stem(
        stem_type=stem_type,
        base_prompt="yeat song",
        duration=2
    )
    stems[stem_type] = audio
    print(f"\n{stem_type}:")
    print(f"  Shape: {audio.shape}")
    print(f"  Has values: {np.any(audio != 0)}")
    if np.any(audio != 0):
        print(f"  Min: {audio.min()}, Max: {audio.max()}, Mean: {audio.mean()}")

print("\nMixing stems...")
mixed = gen.mix_stems(stems, apply_mastering=False)  # No mastering
print(f"\nMixed (no mastering):")
print(f"  Shape: {mixed.shape}")
print(f"  Min: {mixed.min()}, Max: {mixed.max()}, Mean: {mixed.mean()}")

mixed_with_mastering = gen.mix_stems(stems, apply_mastering=True)
print(f"\nMixed (with mastering):")
print(f"  Shape: {mixed_with_mastering.shape}")
print(f"  Min: {mixed_with_mastering.min()}, Max: {mixed_with_mastering.max()}, Mean: {mixed_with_mastering.mean()}")
