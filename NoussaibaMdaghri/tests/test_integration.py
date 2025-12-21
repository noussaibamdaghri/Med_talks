# tests/test_integration.py - Version ULTRA SIMPLE
import sys
import os

# Ajoute le répertoire courant (medical_chatbot) au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import direct depuis les modules
from models import APIResponse
from wikipedia import wikipedia_client
from openfda import openfda_client

class SimpleAPIManager:
    """Manager simple pour tester"""
    def search(self, query: str, sources: list = None):
        response = APIResponse(query)
        
        if sources is None:
            sources = ["wikipedia", "openfda"]
        
        for source in sources:
            if source == "wikipedia":
                try:
                    results = wikipedia_client.search(query, max_results=2)
                    for result in results:
                        response.add_result(result)
                    print(f"  Wikipedia: {len(results)} résultats")
                except Exception as e:
                    response.add_error(f"Wikipedia error: {str(e)}")
                    print(f"  Wikipedia error: {e}")
            
            elif source == "openfda":
                try:
                    # Médicaments
                    drug_results = openfda_client.search_drugs(query, max_results=2)
                    for result in drug_results:
                        response.add_result(result)
                    
                    # Effets secondaires
                    adverse_results = openfda_client.search_adverse_effects(query, max_results=1)
                    for result in adverse_results:
                        response.add_result(result)
                    
                    print(f"  OpenFDA: {len(drug_results) + len(adverse_results)} résultats")
                except Exception as e:
                    response.add_error(f"OpenFDA error: {str(e)}")
                    print(f"  OpenFDA error: {e}")
        
        return response

def test_complete_workflow():
    print("🚀 Test d'intégration complet...")
    print("=" * 50)
    
    manager = SimpleAPIManager()
    
    print("🔍 Recherche: 'aspirine'")
    response = manager.search("aspirine")
    
    print(f"\n📊 Résultats totaux: {len(response.results)}")
    print(f"✅ Succès: {response.success}")
    
    if response.errors:
        print(f"❌ Erreurs: {response.errors}")
    
    # Affiche les résultats par source
    print("\n📚 Résultats par source:")
    sources_count = {}
    for result in response.results:
        sources_count[result.source] = sources_count.get(result.source, 0) + 1
    
    for source, count in sources_count.items():
        print(f"  {source}: {count} résultats")
    
    # Affiche quelques résultats
    if response.results:
        print("\n🎯 Exemple de résultats:")
        for i, result in enumerate(response.results[:3], 1):
            print(f"  {i}. [{result.source}] {result.title}")
            if result.content:
                content_preview = result.content.replace('\n', ' ').strip()
                print(f"     {content_preview[:80]}...")
            print()
    else:
        print("\n⚠️  Aucun résultat trouvé")
    
    print("=" * 50)
    if len(response.results) > 0:
        print("✅ Test d'intégration réussi!")
        return True
    else:
        print("❌ Test d'intégration échoué: aucun résultat")
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)
