"""
Client pour l'API OpenFDA (Food and Drug Administration USA)
Récupère des infos sur les médicaments et effets secondaires
"""
import logging
from typing import List, Optional, Dict, Any
from api_client import http_client
from parsers import cleaner
from models import APIResponse, APIResult

logger = logging.getLogger(__name__)

class OpenFDAClient:
    """
    CLIENT OPENFDA
    Accède aux données officielles des médicaments américains
    """
    
    BASE_URL = "https://api.fda.gov"
    
    def search_drugs(self, drug_name: str, max_results: int = 5) -> List[APIResult]:
        """
        Cherche des médicaments par nom
        
        Args:
            drug_name: Nom du médicament (ex: "aspirin", "paracetamol")
            max_results: Nombre max de résultats
            
        Returns:
            Liste d'informations sur les médicaments
            
        Exemple:
            client.search_drugs("aspirin") → [Infos sur l'aspirine]
        """
        try:
            logger.info(f"💊 OpenFDA search: '{drug_name}'")
            
            # Construction de la requête de recherche
            search_query = f'generic_name:"{drug_name}" OR brand_name:"{drug_name}" OR openfda.substance_name:"{drug_name}"'
            
            params = {
                'search': search_query,
                'limit': max_results,
                'skip': 0
            }
            
            # Appel à l'API OpenFDA
            data = http_client.get(
                f"{self.BASE_URL}/drug/label.json",
                params=params
            )
            
            # Vérifie s'il y a des résultats
            if 'results' not in data:
                logger.warning(f"⚠️  Aucun médicament trouvé pour: {drug_name}")
                return []
            
            results = []
            drugs = data['results']
            
            logger.info(f"💊 OpenFDA a trouvé {len(drugs)} médicaments")
            
            # Parse chaque médicament
            for drug in drugs[:max_results]:
                result = self._parse_drug(drug, drug_name)
                if result:
                    results.append(result)
                    logger.debug(f"  ✓ {result.title}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur OpenFDA drugs: {str(e)}")
            return []
    
    def search_adverse_effects(self, drug_name: str, max_results: int = 3) -> List[APIResult]:
        """
        Cherche les effets secondaires rapportés pour un médicament
        
        Args:
            drug_name: Nom du médicament
            max_results: Nombre max de rapports
            
        Returns:
            Liste d'effets secondaires
        """
        try:
            logger.info(f"⚠️  Recherche effets secondaires pour: '{drug_name}'")
            
            params = {
                'search': f'patient.drug.medicinalproduct:"{drug_name}"',
                'limit': max_results,
                'skip': 0
            }
            
            data = http_client.get(
                f"{self.BASE_URL}/drug/event.json",
                params=params
            )
            
            if 'results' not in data:
                return []
            
            results = []
            for event in data['results'][:max_results]:
                result = self._parse_adverse_event(event, drug_name)
                if result:
                    results.append(result)
            
            logger.info(f"⚠️  Trouvé {len(results)} rapports d'effets secondaires")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur OpenFDA adverse effects: {str(e)}")
            return []
    
    def _parse_drug(self, drug_data: Dict[str, Any], query: str) -> Optional[APIResult]:
      """Transforme les données brutes d'un médicament en APIResult"""
      try:
            openfda_info = drug_data.get('openfda', {})
        
            brand_names = openfda_info.get('brand_name', [])
            generic_names = openfda_info.get('generic_name', [])
        
            brand_name = brand_names[0] if brand_names else query.capitalize()
            generic_name = generic_names[0] if generic_names else query
        
            title = f"{brand_name} ({generic_name})" if brand_name != generic_name else generic_name
            
            # Récupère les sections importantes
            indications = drug_data.get('indications_and_usage', [''])
            dosage = drug_data.get('dosage_and_administration', [''])
            warnings = drug_data.get('warnings', [''])
            description = drug_data.get('description', [''])
            
            # Nettoie chaque section
            clean_indications = cleaner.clean_html(indications[0] if indications else '', 300)
            clean_dosage = cleaner.clean_html(dosage[0] if dosage else '', 200)
            clean_warnings = cleaner.clean_html(warnings[0] if warnings else '', 200)
            clean_description = cleaner.clean_html(description[0] if description else '', 400)
            
            # Construit le contenu final
            content_parts = []
            
            if clean_description:
                content_parts.append(f"📋 Description: {clean_description}")
            
            if clean_indications:
                content_parts.append(f"🎯 Indications: {clean_indications}")
            
            if clean_dosage:
                content_parts.append(f"💊 Dosage: {clean_dosage}")
            
            if clean_warnings:
                content_parts.append(f"⚠️  Avertissements: {clean_warnings}")
            
            content = "\n\n".join(content_parts)
            
            spl_ids = openfda_info.get('spl_id', [])
            url = None
            if spl_ids:
             spl_id = spl_ids[0]
             url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={spl_id}"
        
            manufacturer = openfda_info.get('manufacturer_name', ['Inconnu'])[0]
        
            return APIResult(
              source="openfda",  # ← Spécifie la source ici
              title=title,
              content=content,
              url=url,
              confidence=0.95,
              metadata={  # ← Tout dans metadata
                'drug_id': spl_ids[0] if spl_ids else None,
                'manufacturer': manufacturer,
                'brand_names': brand_names,
                'generic_names': generic_names,
                'route': openfda_info.get('route', [''])[0],
                'product_type': openfda_info.get('product_type', [''])[0]
            }
        )
        
      except Exception as e:
         logger.error(f"❌ Erreur parsing médicament OpenFDA: {str(e)}")
         return None
    
    def _parse_adverse_event(self, event_data: Dict[str, Any], drug_name: str) -> Optional[APIResult]:
        """
        Parse un rapport d'effet secondaire
        """
        try:
            # Récupère les réactions du patient
            patient = event_data.get('patient', {})
            reactions = patient.get('reaction', [])
            
            if not reactions:
                return None
            
            # Liste des réactions (max 5)
            reaction_list = []
            for reaction in reactions[:5]:
                if 'reactionmeddrapt' in reaction:
                    reaction_list.append(reaction['reactionmeddrapt'])
            
            if not reaction_list:
                return None
            
            # Gravité
            seriousness = event_data.get('serious', 'Non spécifié')
            seriousness_fr = {
                '1': 'Non grave',
                '2': 'Grave',
                '': 'Non spécifié'
            }.get(str(seriousness), str(seriousness))
            
            # Construit le contenu
            content_parts = [
                f"💊 Médicament: {drug_name}",
                f"⚠️  Réactions rapportées:",
                "\n".join([f"   • {r}" for r in reaction_list]),
                f"📊 Gravité: {seriousness_fr}"
            ]
            
            # Infos patient si disponibles
            if 'patientonsetage' in patient:
                content_parts.append(f"👤 Âge patient: {patient['patientonsetage']} ans")
            
            if 'patientsex' in patient:
                sex_map = {'1': 'Homme', '2': 'Femme', '0': 'Non spécifié'}
                sex = sex_map.get(str(patient['patientsex']), 'Non spécifié')
                content_parts.append(f"⚤ Sexe: {sex}")
            
            content = "\n".join(content_parts)
            
            return APIResult(
                source="openfda",
                title=f"Effets secondaires - {drug_name}",
                content=content,
                url=None,
                confidence=0.7,  # Données de rapports, à interpréter
                metadata={
                    'reactions': reaction_list,
                    'seriousness': seriousness,
                    'report_date': event_data.get('receivedate', '')
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur parsing effet secondaire: {str(e)}")
            return None

# Instance globale
openfda_client = OpenFDAClient()