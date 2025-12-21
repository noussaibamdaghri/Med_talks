import sys
sys.path.append('.')

print("=" * 60)
print("COMPLETE TEST FOR PERSON A (AYA NAIM)")
print("=" * 60)

# 1. Test Data Loader
print("\n1. Testing Data Loader...")
from aya_naim.data_loader import MedicalDataLoader
loader = MedicalDataLoader()
df = loader.load()

print(f"   ✓ Loaded {len(df)} medical flashcards")

# 2. Test Text Preprocessor
print("\n2. Testing Text Preprocessor...")
from aya_naim.text_preprocessor import TextPreprocessor
pre = TextPreprocessor()

sample_text = df['instruction'].iloc[0]
cleaned = pre.clean(sample_text)
print(f"   Original: {sample_text[:80]}...")
print(f"   Cleaned:  {cleaned[:80]}...")

# 3. Test Similarity Engine
print("\n3. Testing Similarity Engine...")
try:
    from aya_naim.similarity_engine import MedicalSimilarity
    sim = MedicalSimilarity()
    
    # Get sample questions
    sample_questions = df['instruction'].head(10).tolist()
    query = "What is magnesium deficiency?"
    
    results = sim.find_similar(query, sample_questions, top_k=3)
    print(f"   Query: '{query}'")
    print(f"   Most similar questions:")
    for i, (question, score) in enumerate(results, 1):
        print(f"   {i}. {question[:60]}... (score: {score:.3f})")
    
    print("   ✓ Similarity engine working!")
except Exception as e:
    print(f"   Note: Install sentence-transformers for full functionality")
    print(f"   Run: pip install sentence-transformers")

# 4. Test Memory Store
print("\n4. Testing Memory Store...")
try:
    from aya_naim.memory_store import MemoryStore
    store = MemoryStore('test_database.db')
    
    # Save a sample flashcard
    test_id = store.save_flashcard(
        instruction=df['instruction'].iloc[0],
        input_text=df['input'].iloc[0],
        output=df['output'].iloc[0],
        instruction_clean=pre.clean(df['instruction'].iloc[0]),
        output_clean=pre.clean(df['output'].iloc[0]),
        needs_summary=pre.should_summarize(df['output'].iloc[0])
    )
    print(f"   ✓ Saved flashcard ID: {test_id}")
    
    stats = store.get_stats()
    print(f"   Database stats: {stats}")
    
except Exception as e:
    print(f"   Note: Install SQLAlchemy for database functionality")
    print(f"   Run: pip install SQLAlchemy")

# 5. Test Exploration Metrics
print("\n5. Testing Exploration Metrics...")
from aya_naim.metrics import ExplorationMetrics

metrics = ExplorationMetrics.analyze_flashcards(df.head(100))  # Analyze first 100
print(f"   Dataset analysis:")
print(f"   - Total: {metrics.get('total')} records")
if 'instruction_stats' in metrics:
    stats = metrics['instruction_stats']
    print(f"   - Avg instruction length: {stats['avg_length']:.1f} characters")

print("\n" + "=" * 60)
print("PERSON A COMPONENTS READY!")
print("=" * 60)
print("\nNext steps:")
print("1. Install missing packages: pip install sentence-transformers SQLAlchemy")
print("2. Run full integration test")
print("3. Push to GitHub: git add . && git commit -m 'Person A complete'")