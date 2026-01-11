#!/usr/bin/env python3
"""Battle-tested integration tests for the multi-stem pipeline."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.services.multi_stem_generator import MultiStemGenerator


class BattlePipelineTests(unittest.TestCase):
    """Integration tests that stress the pipeline without heavy models."""

    def setUp(self):
        """Prepare temporary output directory and patch heavy loaders."""
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # Skip weight loading and vocal model initialization to keep tests fast
        self._load_models_patcher = patch.object(
            MultiStemGenerator,
            "_load_models",
            new=lambda self: None
        )
        self._vocal_init_patcher = patch.object(
            MultiStemGenerator,
            "_init_vocal_generator",
            new=lambda self: None
        )
        self._load_models_patcher.start()
        self._vocal_init_patcher.start()
        self.addCleanup(self._load_models_patcher.stop)
        self.addCleanup(self._vocal_init_patcher.stop)

    def _fake_generate_stem(
        self,
        stem_type: str,
        base_prompt: str,
        duration: int,
        **_: dict
    ) -> np.ndarray:
        """Return deterministic audio buffers per stem type."""
        length = max(1, int(duration * 32000))
        if stem_type == "drums":
            wave = np.tile([0.35, -0.35], length // 2 + 1)[:length]
        elif stem_type == "bass":
            wave = np.linspace(-0.4, 0.4, length)
        elif stem_type == "melody_1":
            wave = np.sin(np.linspace(0, np.pi * 4, length))
        elif stem_type == "melody_2":
            wave = np.cos(np.linspace(0, np.pi * 6, length))
        elif stem_type == "vocals":
            wave = np.linspace(-0.2, 0.2, length)
        else:
            wave = np.zeros(length)
        return wave.astype(np.float32)

    def test_generate_all_stems_to_disk(self):
        """MultiStemGenerator.generate writes valid stem and mix files."""
        generator = MultiStemGenerator(enable_processing=True, enable_mastering=True)

        with patch.object(MultiStemGenerator, "generate_stem", new=self._fake_generate_stem):
            mixed_path, stem_paths = generator.generate(
                prompt="battle test prompt",
                duration=4,
                stems_to_generate=["drums", "bass", "melody_1", "melody_2"],
                output_dir=self.temp_dir.name,
                save_individual_stems=True
            )

        # Ensure files exist and are readable via soundfile
        for stem_file in stem_paths.values():
            audio, sr = sf.read(stem_file)
            self.assertEqual(sr, 32000)
            self.assertGreater(audio.size, 0)

        mixed_audio, mixed_sr = sf.read(mixed_path)
        self.assertEqual(mixed_sr, 32000)
        self.assertGreater(mixed_audio.size, 0)

    def test_generate_sections_crossfade(self):
        """generate_sections produces mastered audio with crossfades."""
        generator = MultiStemGenerator(enable_processing=True, enable_mastering=True)

        with patch.object(MultiStemGenerator, "generate_stem", new=self._fake_generate_stem):
            output_file, section_audio = generator.generate_sections(
                lyrics_sections=[
                    ("intro", "Introduce the vibe"),
                    ("chorus", "Catchy hook lyrics"),
                    ("verse", "Detailed verse content")
                ],
                base_prompt="battle prompt",
                stems_to_generate=["drums", "bass"],
                output_dir=self.temp_dir.name,
                crossfade_duration=0.25
            )

        rendered, sr = sf.read(output_file)
        self.assertEqual(sr, 32000)
        self.assertGreater(rendered.size, 0)
        self.assertEqual(len(section_audio), 3)

    def test_prepare_audio_for_export_handles_shapes(self):
        """_prepare_audio_for_export normalizes multiple array shapes."""
        generator = MultiStemGenerator()

        mono = generator._prepare_audio_for_export(np.zeros((16000,), dtype=np.float32))
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(mono.shape[0], 16000)

        stereo = generator._prepare_audio_for_export(np.zeros((2, 8000), dtype=np.float32))
        self.assertEqual(stereo.shape, (8000, 2))

        empty = generator._prepare_audio_for_export(np.array([]))
        self.assertEqual(empty.ndim, 1)
        self.assertEqual(empty.shape[0], 1)

        scalar = generator._prepare_audio_for_export(np.array(0.0))
        self.assertEqual(scalar.ndim, 1)
        self.assertEqual(scalar.shape[0], 1)

    def test_mix_stems_balanced_levels(self):
        """mix_stems returns finite audio with custom levels."""
        generator = MultiStemGenerator()
        sample = np.linspace(-1.0, 1.0, 64000, dtype=np.float32)
        stems = {
            "drums": sample,
            "bass": sample * 0.5,
            "melody_1": sample * 0.3
        }
        mixed = generator.mix_stems(stems, mix_levels={"drums": 1.0, "bass": 0.8, "melody_1": 0.6})
        self.assertFalse(np.isnan(mixed).any())
        self.assertFalse(np.isinf(mixed).any())
        self.assertGreater(mixed.size, 0)


if __name__ == "__main__":
    unittest.main()
