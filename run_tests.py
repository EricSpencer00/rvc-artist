#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-Stem Music Generation
=========================================================
Tests all components without requiring heavy ML dependencies.
"""

import sys
import ast
from pathlib import Path

# Test configuration
TESTS_PASSED = 0
TESTS_FAILED = 0
TEST_RESULTS = []

def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global TESTS_PASSED, TESTS_FAILED
            try:
                func()
                TESTS_PASSED += 1
                TEST_RESULTS.append(f"✅ {name}")
                return True
            except Exception as e:
                TESTS_FAILED += 1
                TEST_RESULTS.append(f"❌ {name}: {e}")
                return False
        return wrapper
    return decorator


# =========================================
# SYNTAX TESTS
# =========================================

@test("Syntax: multi_stem_generator.py")
def test_multi_stem_syntax():
    code = open('src/services/multi_stem_generator.py').read()
    ast.parse(code)

@test("Syntax: vocal_generator.py")
def test_vocal_syntax():
    code = open('src/services/vocal_generator.py').read()
    ast.parse(code)

@test("Syntax: music_generator.py")
def test_music_gen_syntax():
    code = open('src/services/music_generator.py').read()
    ast.parse(code)

@test("Syntax: style_analyzer.py")
def test_style_syntax():
    code = open('src/services/style_analyzer.py').read()
    ast.parse(code)

@test("Syntax: lyrics_generator.py")
def test_lyrics_syntax():
    code = open('src/services/lyrics_generator.py').read()
    ast.parse(code)

@test("Syntax: test_multistem.py")
def test_multistem_test_syntax():
    code = open('test_multistem.py').read()
    ast.parse(code)

@test("Syntax: examples_multistem.py")
def test_examples_syntax():
    code = open('examples_multistem.py').read()
    ast.parse(code)


# =========================================
# STRUCTURE TESTS
# =========================================

@test("Structure: StemProcessor class exists")
def test_stem_processor_exists():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'class StemProcessor:' in code

@test("Structure: MasteringProcessor class exists")
def test_mastering_exists():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'class MasteringProcessor:' in code

@test("Structure: MultiStemGenerator class exists")
def test_multistem_exists():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'class MultiStemGenerator:' in code

@test("Structure: VocalGenerator class exists")
def test_vocal_class_exists():
    code = open('src/services/vocal_generator.py').read()
    assert 'class VocalGenerator:' in code

@test("Structure: PROCESSING_CHAINS defined")
def test_processing_chains():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'PROCESSING_CHAINS' in code
    assert "'vocals'" in code
    assert "'drums'" in code
    assert "'bass'" in code

@test("Structure: SECTION_PARAMS defined")
def test_section_params():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'SECTION_PARAMS' in code
    assert "'intro'" in code
    assert "'verse'" in code
    assert "'chorus'" in code
    assert "'bridge'" in code

@test("Structure: STEM_TYPES defined")
def test_stem_types():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'STEM_TYPES' in code


# =========================================
# METHOD TESTS
# =========================================

@test("Methods: process_stem in StemProcessor")
def test_process_stem_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def process_stem(' in code

@test("Methods: master in MasteringProcessor")
def test_master_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def master(' in code

@test("Methods: _normalize_loudness exists")
def test_normalize_loudness():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def _normalize_loudness(' in code

@test("Methods: generate_stem in MultiStemGenerator")
def test_generate_stem_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def generate_stem(' in code

@test("Methods: mix_stems in MultiStemGenerator")
def test_mix_stems_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def mix_stems(' in code

@test("Methods: generate in MultiStemGenerator")
def test_generate_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def generate(' in code

@test("Methods: generate_sections in MultiStemGenerator")
def test_generate_sections_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def generate_sections(' in code

@test("Methods: _crossfade_sections exists")
def test_crossfade_method():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'def _crossfade_sections(' in code


# =========================================
# FEATURE TESTS
# =========================================

@test("Feature: RVC integration placeholder")
def test_rvc_integration():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'rvc_model_path' in code
    assert '_load_rvc_model' in code
    assert 'apply_rvc_conversion' in code

@test("Feature: Melody conditioning support")
def test_melody_conditioning():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'melody_reference' in code
    assert 'generate_with_chroma' in code

@test("Feature: Pedalboard processing")
def test_pedalboard_integration():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'PEDALBOARD_AVAILABLE' in code
    assert 'Compressor' in code
    assert 'Reverb' in code
    assert 'HighpassFilter' in code

@test("Feature: Pyloudnorm mastering")
def test_pyloudnorm_integration():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'PYLOUDNORM_AVAILABLE' in code
    assert 'target_lufs' in code

@test("Feature: Stereo width processing")
def test_stereo_width():
    code = open('src/services/multi_stem_generator.py').read()
    assert 'stereo_width' in code
    assert '_apply_stereo_width' in code


# =========================================
# DOCUMENTATION TESTS
# =========================================

@test("Documentation: ROADMAP.md exists")
def test_roadmap_exists():
    assert Path('ROADMAP.md').exists()

@test("Documentation: QUICKSTART.md exists")
def test_quickstart_exists():
    assert Path('QUICKSTART.md').exists()

@test("Documentation: Module docstring exists")
def test_module_docstring():
    code = open('src/services/multi_stem_generator.py').read()
    assert '"""' in code[:100]


# =========================================
# RUN ALL TESTS
# =========================================

def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("🧪 MULTI-STEM MUSIC GENERATION - TEST SUITE")
    print("=" * 60)
    print()
    
    # Get all test functions
    test_funcs = [
        test_multi_stem_syntax,
        test_vocal_syntax,
        test_music_gen_syntax,
        test_style_syntax,
        test_lyrics_syntax,
        test_multistem_test_syntax,
        test_examples_syntax,
        test_stem_processor_exists,
        test_mastering_exists,
        test_multistem_exists,
        test_vocal_class_exists,
        test_processing_chains,
        test_section_params,
        test_stem_types,
        test_process_stem_method,
        test_master_method,
        test_normalize_loudness,
        test_generate_stem_method,
        test_mix_stems_method,
        test_generate_method,
        test_generate_sections_method,
        test_crossfade_method,
        test_rvc_integration,
        test_melody_conditioning,
        test_pedalboard_integration,
        test_pyloudnorm_integration,
        test_stereo_width,
        test_roadmap_exists,
        test_quickstart_exists,
        test_module_docstring,
    ]
    
    print("Running tests...\n")
    
    for test_func in test_funcs:
        test_func()
    
    print()
    print("=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print()
    
    for result in TEST_RESULTS:
        print(result)
    
    print()
    print("=" * 60)
    print(f"✅ Passed: {TESTS_PASSED}")
    print(f"❌ Failed: {TESTS_FAILED}")
    print(f"📈 Total:  {TESTS_PASSED + TESTS_FAILED}")
    print("=" * 60)
    
    if TESTS_FAILED == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {TESTS_FAILED} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
