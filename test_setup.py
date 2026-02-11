#!/usr/bin/env python3
"""Test script to verify installation and setup.

Run this before running the full analysis to check:
- Dependencies are installed
- Config file is valid
- Data files are accessible
- Module imports work
"""

import sys
from pathlib import Path

print("Immigration Location Analysis - Setup Test")
print("=" * 60)
print()

# Test 1: Python version
print("1. Checking Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")
if version.major < 3 or (version.major == 3 and version.minor < 12):
    print("   ⚠️  Warning: Python 3.12+ recommended")
else:
    print("   ✓ Version OK")
print()

# Test 2: Dependencies
print("2. Checking dependencies...")
dependencies = [
    'pandas',
    'numpy',
    'yaml',
    'geopy',
    'matplotlib',
    'seaborn',
    'plotly',
    'folium'
]

missing = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f"   ✓ {dep}")
    except ImportError:
        print(f"   ✗ {dep} - MISSING")
        missing.append(dep)

if missing:
    print()
    print(f"   Missing dependencies: {', '.join(missing)}")
    print("   Install with: pip install -r requirements.txt")
    print()
else:
    print("   ✓ All dependencies installed")
print()

# Test 3: Config file
print("3. Checking configuration...")
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists():
    print(f"   ✓ Config file found: {config_path}")
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Config is valid YAML")
        print(f"   Start date: {config.get('start_date', 'NOT SET')}")
        print(f"   Partner 1 files: {len(config.get('data_files', {}).get('partner1', []))}")
        print(f"   Partner 2 files: {len(config.get('data_files', {}).get('partner2', []))}")
    except Exception as e:
        print(f"   ✗ Config error: {e}")
else:
    print(f"   ✗ Config file not found: {config_path}")
print()

# Test 4: Data directory
print("4. Checking data directory...")
data_path = Path(__file__).parent / "data"
if data_path.exists():
    print(f"   ✓ Data directory exists: {data_path}")

    # List JSON files
    json_files = list(data_path.glob("*.json"))
    print(f"   Found {len(json_files)} JSON files:")
    for json_file in json_files:
        size_mb = json_file.stat().st_size / 1024 / 1024
        # Check if symlink
        if json_file.is_symlink():
            target = json_file.resolve()
            print(f"     - {json_file.name} -> {target.name} ({size_mb:.1f} MB)")
        else:
            print(f"     - {json_file.name} ({size_mb:.1f} MB)")
else:
    print(f"   ✗ Data directory not found: {data_path}")
    print("   Create with: mkdir -p data")
print()

# Test 5: Module imports
print("5. Checking module imports...")
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from analyzer import LocationAnalyzer
    print("   ✓ LocationAnalyzer")
except ImportError as e:
    print(f"   ✗ LocationAnalyzer - {e}")

try:
    from interpolator import LocationInterpolator
    print("   ✓ LocationInterpolator")
except ImportError as e:
    print(f"   ✗ LocationInterpolator - {e}")

try:
    from visualizer import LocationVisualizer
    print("   ✓ LocationVisualizer")
except ImportError as e:
    print(f"   ✗ LocationVisualizer - {e}")

try:
    from report import INDReportGenerator
    print("   ✓ INDReportGenerator")
except ImportError as e:
    print(f"   ✗ INDReportGenerator - {e}")
print()

# Test 6: Output directory
print("6. Checking output directory...")
output_path = Path(__file__).parent / "output"
if output_path.exists():
    print(f"   ✓ Output directory exists: {output_path}")
else:
    print(f"   Creating output directory: {output_path}")
    output_path.mkdir(parents=True)
    print("   ✓ Created")
print()

# Summary
print("=" * 60)
print("Setup Test Complete")
print("=" * 60)

if missing:
    print()
    print("⚠️  Some dependencies are missing. Install them first:")
    print("   pip install -r requirements.txt")
    print()
    sys.exit(1)
else:
    print()
    print("✓ Setup looks good! You can run the analysis:")
    print("   python run.py")
    print()
    print("   OR")
    print()
    print("   ./run.py")
    print()
    sys.exit(0)
