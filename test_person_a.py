# TEST FOR PERSON A (AYA NAIM)
import sys
sys.path.append('.')

print("=" * 50)
print("TESTING PERSON A COMPONENTS")
print("=" * 50)

# Test 1: Data Loader
print("\n1. Testing Data Loader...")
from aya_naim.data_loader import MedicalDataLoader

loader = MedicalDataLoader()
df = loader.load()

if df is not None:
    print(f"   ✓ SUCCESS: {len(df)} records")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    info = loader.info()
    print(f"\n   Sample record:")
    for key, value in info['sample'].items():
        if isinstance(value, str) and len(value) > 60:
            print(f"   - {key}: {value[:60]}...")
        else:
            print(f"   - {key}: {value}")
else:
    print("   ✗ FAILED to load data")

# Test 2: Text Preprocessor
print("\n2. Testing Text Preprocessor...")
from aya_naim.text_preprocessor import TextPreprocessor

processor = TextPreprocessor()
test_text = "  Patient: 'Severe headache!!!' (age 45)  "
cleaned = processor.clean(test_text)

print(f"   Original: {test_text}")
print(f"   Cleaned:  {cleaned}")

# Test summary check
long_text = " ".join(["word" for _ in range(200)])
print(f"\n   Summary check:")
print(f"   - Long text needs summary: {processor.should_summarize(long_text)}")
print(f"   - Short text needs summary: {processor.should_summarize('short')}")

print("\n" + "=" * 50)
print("PERSON A SETUP COMPLETE!")
print("=" * 50)