import os
import sys
from typing import Dict, Any

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

try:
    sys.path.insert(0, 'aya_sindel')
    from aya_sindel.handler import LLMHandler
except ImportError as e:
    print(f"Erreur import aya_sindel: {e}")
    print("Assure-toi que le dossier 'aya_sindel' existe")
    sys.exit(1)

try:
    sys.path.insert(0, 'elaazaouzi_fadwa')
    from elaazaouzi_fadwa.orchestrateur import PersonBOrchestrator
except ImportError:
    print("elaazaouzi_fadwa non trouvé, utilisation de mock")
    
    def classify_question(question):
        """Mock de classification"""
        question_lower = question.lower()
        if "?" in question and ("or" in question_lower or "which" in question_lower):
            return "qcm"
        elif "how to" in question_lower or "step" in question_lower:
            return "stepwise"
        elif "what is" in question_lower or "define" in question_lower:
            return "definition"
        else:
            return "definition"

try:
    sys.path.insert(0, 'aya_naim')
    from aya_naim.orchestrator import PersonAOrchestrator
except ImportError:
    print("aya_naim non trouvé, utilisation de mock")
    
    def search_dataset(question):
        """Mock de recherche dataset"""
        medical_info = {
            "diabetes": "Diabetes is a chronic condition with high blood sugar levels.",
            "hypertension": "Hypertension is high blood pressure >140/90 mmHg.",
            "asthma": "Asthma is a lung condition causing breathing difficulties.",
            "cpr": "CPR involves chest compressions and rescue breathing."
        }
        
        for keyword, info in medical_info.items():
            if keyword in question.lower():
                return info
        
        return f"Medical information from dataset about: {question}"

try:
    sys.path.insert(0, 'noussaiba_mdaghri')
    from noussaiba_mdaghri.orchestrator import PersonCOrchestrator
except ImportError:
    print("noussaiba_mdaghri non trouvé, utilisation de mock")
    
    def search_apis(question):
        """Mock de recherche API"""
        return f"Information from external medical APIs about: {question}"

class Orchestrateur:
    """Orchestrateur principal qui connecte toutes les personnes"""
    
    def __init__(self):
        print("\nInitialisation de l'orchestrateur...")
        
        self.aya_sindel = LLMHandler()
        
        stats = self.aya_sindel.get_stats()
        mode = stats['llm']['mode']
        print(f"Mode : {mode.upper()}")
        
        if mode != 'production':
            print("ATTENTION:  en mode développement")
            print("Les réponses seront mockées, pas générées par GPT")
        
        print("Orchestrateur initialisé")
    
    def process_question(self, user_question: str) -> Dict[str, Any]:

        print(f"\n{'='*60}")
        print(f"QUESTION UTILISATEUR: {user_question}")
        print(f"{'='*60}")
        
        print("\n1.Classification de la question...")
        question_type = classify_question(user_question)
        print(f"Type détecté: {question_type}")
        
        print("\n2.Recherche dans le dataset médical...")
        dataset_info = search_dataset(user_question)
        print(f"Dataset trouvé: {len(dataset_info)} caractères")
        
        print("\n3.Recherche via APIs externes...")
        api_info = search_apis(user_question)
        print(f"API info trouvée: {len(api_info)} caractères")
        
        print(f"\n4.Génération de la réponse ({question_type})...")
        
        result = self.aya_sindel.process_question(
            question=user_question,
            question_type=question_type,
            dataset_info=dataset_info,
            api_info=api_info
        )
        
        print("\nRéponse générée")
        print("="*60)
        
        return {
            "answer": result["answer"],
            "question_type": question_type,
            "dataset_length": len(dataset_info),
            "api_length": len(api_info),
            "person_d_metadata": result.get("metadata", {}),
            "status": result.get("status", "unknown"),
            "full_pipeline": True
        }
    
    def run_test_suite(self):
        """Exécute une suite de tests complets"""
        print("\n" + "="*60)
        print("SUITE DE TESTS D'INTÉGRATION COMPLÈTE")
        print("="*60)
        
        test_questions = [
            "What is diabetes mellitus and what are its main types?",
            "What are the diagnostic criteria for hypertension?"
        ]
        
        results = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nTEST {i}/{len(test_questions)}")
            print(f"Question: {question[:80]}...")
            
            try:
                result = self.process_question(question)
                results.append(result)
                
                print("\nRésumé Test:")
                print(f"Type: {result['question_type']}")
                print(f"Status: {result['status']}")
                print(f"Longueur réponse: {len(result['answer'])} caractères")
                
                print("\nExtrait réponse:")
                print("-" * 40)
                print(result['answer'][:200] + ("..." if len(result['answer']) > 200 else ""))
                print("-" * 40)
                
            except Exception as e:
                print(f"Erreur pendant le test {i}: {e}")
                import traceback
                traceback.print_exc()
                results.append({"error": str(e)})
        
        print("\n" + "="*60)
        print("STATISTIQUES FINALES")
        print("="*60)
        
        successful = sum(1 for r in results if "answer" in r)
        print(f"Tests réussis: {successful}/{len(test_questions)}")
        
        if successful > 0:
            total_chars = sum(len(r.get('answer', '')) for r in results)
            avg_chars = total_chars / successful
            print(f"Longueur moyenne réponse: {avg_chars:.0f} caractères")
        
        print("\n" + "="*60)
        if successful == len(test_questions):
            print("TOUS LES TESTS ONT RÉUSSI")
        else:
            print(f"{len(test_questions) - successful} test(s) ont échoué")
        print("="*60)

def main():
    print("\nORCHESTRATEUR PRINCIPAL DU PROJET MÉDICAL")
    print("=" * 60)
    
    try:
        orchestrateur = Orchestrateur()
        
        print("\nChoisis une option:")
        print("1. Poser une question spécifique")
        print("2. Exécuter la suite de tests complète")
        print("3. Quitter")
        
        choice = input("\nTon choix (1-3): ").strip()
        
        if choice == "1":
            question = input("\nPose ta question médicale: ").strip()
            if question:
                result = orchestrateur.process_question(question)
                print(f"\n{'='*60}")
                print("RÉPONSE FINALE:")
                print("="*60)
                print(result["answer"])
                print("="*60)
                
                print("\nMétadonnées:")
                print(f"- Type de question: {result['question_type']}")
                print(f"- Longueur: {len(result['answer'])} caractères")
                print(f"- Status: {result['status']}")
        
        elif choice == "2":
            orchestrateur.run_test_suite()
        
        else:
            print("\nAu revoir")
    
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur")
    except Exception as e:
        print(f"\nErreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
