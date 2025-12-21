import sys
import os

# Ajoute le répertoire parent (medical_chatbot) au Python Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Maintenant importe directement
from wikipedia import wikipedia_client

def test_wikipedia_search():
    print("🧪 Test Wikipedia search...")
    results = wikipedia_client.search("aspirine", max_results=2)
    
    print(f"  Nombre de résultats: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.title}")
        print(f"     {result.content[:100]}...")
        print()
    
    assert len(results) > 0, "Devrait trouver des résultats"
    print("✅ Test Wikipedia réussi!")

if __name__ == "__main__":
    test_wikipedia_search()