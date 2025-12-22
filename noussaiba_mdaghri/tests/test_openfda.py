# tests/test_openfda.py - Version corrigée
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openfda import openfda_client

def test_openfda_drug_search():
    print("🧪 Test OpenFDA drug search...")
    results = openfda_client.search_drugs("aspirin", max_results=2)
    
    print(f"  Médicaments trouvés: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.title}")
        
        # ❌ AVANT : result.manufacturer (n'existe plus)
        # ✅ APRÈS : result.metadata.get('manufacturer')
        manufacturer = result.metadata.get('manufacturer', 'Inconnu')
        print(f"     Fabricant: {manufacturer}")
        
        # Affiche un aperçu du contenu
        if result.content:
            preview = result.content.replace('\n', ' ').strip()
            print(f"     Contenu: {preview[:100]}...")
        print()
    
    assert len(results) > 0, "Devrait trouver des médicaments"
    print("✅ Test OpenFDA réussi!")

def test_openfda_adverse_effects():
    print("🧪 Test OpenFDA adverse effects...")
    results = openfda_client.search_adverse_effects("aspirin", max_results=1)
    
    print(f"  Rapports d'effets secondaires: {len(results)}")
    
    if results:
        result = results[0]
        print(f"  {result.title}")
        print(f"  {result.content[:150]}...")
        print(f"  Réactions: {result.metadata.get('reactions', [])}")
    
    print("✅ Test effets secondaires terminé!")

if __name__ == "__main__":
    test_openfda_drug_search()
    print()
    test_openfda_adverse_effects()
