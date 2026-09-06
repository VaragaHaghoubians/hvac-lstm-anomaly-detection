#!/usr/bin/env python3
"""
Quick Test Script - Verify Modular LSTM Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run this before executing main.py to verify all dependencies
and file structure are correct.

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*70)
    print("🧪 TESTING PACKAGE IMPORTS")
    print("="*70 + "\n")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('configparser', 'ConfigParser')
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name} installed")
        except ImportError:
            print(f"   ❌ {name} NOT installed")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Install missing packages with:")
        print("   pip install -r requirements.txt\n")
        return False
    
    print("\n✅ All packages installed!\n")
    return True


def test_file_structure():
    """Test if all required files exist"""
    print("="*70)
    print("📁 TESTING FILE STRUCTURE")
    print("="*70 + "\n")
    
    script_dir = Path(__file__).parent
    
    required_files = [
        '1_main.py',
        '2_data_loader.py',
        '3_preprocessor.py',
        '4_model.py',
        '5_trainer.py',
        '6_evaluator.py',
        '7_visualizer.py',
        '8_report_generator.py',
        '0_test_setup.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    for filename in required_files:
        filepath = script_dir / filename
        if filepath.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} NOT FOUND")
            all_ok = False
    
    # Check main config.ini in project root
    main_config = script_dir.parent.parent / 'config.ini'
    if main_config.exists():
        print(f"   ✅ ../../config.ini (main config)")
    else:
        print(f"   ❌ ../../config.ini NOT FOUND")
        all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some files missing! Re-run setup.\n")
        return False
    
    print("\n✅ All files present!\n")
    return True


def test_config():
    """Test if config.ini is readable"""
    print("="*70)
    print("⚙️  TESTING CONFIGURATION")
    print("="*70 + "\n")
    
    import configparser
    
    try:
        config = configparser.ConfigParser()
        # Use main config.ini from project root (2 levels up)
        config_path = Path(__file__).parent.parent.parent / 'config.ini'
        config.read(config_path, encoding='utf-8-sig')
        
        # Check required sections
        required_sections = ['global', 'paths', 'lstm_autoencoder']
        for section in required_sections:
            if config.has_section(section):
                print(f"   ✅ [{section}] section found")
            else:
                print(f"   ❌ [{section}] section MISSING")
                return False
        
        # Show key settings
        print(f"\n   📊 KEY SETTINGS:")
        print(f"      • Building: {config.get('global', 'building_id')}")
        print(f"      • Unit: {config.get('global', 'ahu_unit')}")
        print(f"      • Season: {config.get('global', 'season')} {config.get('global', 'year')}")
        print(f"      • Sequence length: {config.get('lstm_autoencoder', 'sequence_length')}")
        print(f"      • Encoder units: {config.get('lstm_autoencoder', 'encoder_units', fallback='96')}")
        print(f"      • Epochs: {config.get('lstm_autoencoder', 'epochs', fallback='250')}")
        print(f"      • Threshold percentile: {config.get('lstm_autoencoder', 'threshold_percentile', fallback='99.5')}")
        
        print("\n✅ Configuration valid!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Config error: {e}\n")
        return False


def test_module_imports():
    """Test if custom modules can be imported"""
    print("="*70)
    print("🔧 TESTING CUSTOM MODULE IMPORTS")
    print("="*70 + "\n")
    
    import importlib
    
    modules = [
        '2_data_loader',
        '3_preprocessor',
        '4_model',
        '5_trainer',
        '6_evaluator',
        '7_visualizer',
        '8_report_generator'
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {module_name}.py imports correctly")
        except Exception as e:
            print(f"   ❌ {module_name}.py import failed: {e}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some modules have import errors!\n")
        return False
    
    print("\n✅ All modules import successfully!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("="*70)
    print("║" + " "*68 + "║")
    print("║" + "🧪 LSTM AUTOENCODER SETUP VERIFICATION".center(68) + "║")
    print("║" + " "*68 + "║")
    print("="*70)
    print("="*70 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Configuration", test_config),
        ("Module Imports", test_module_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("="*70)
    print("📋 SUMMARY")
    print("="*70 + "\n")
    
    all_passed = all(result for _, result in results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")
        print("🚀 You're ready to run:")
        print("   python main.py\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70 + "\n")
        print("⚠️  Fix the issues above before running main.py\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
