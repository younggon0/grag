"""
Graph Extractor Module
Extracts entities and relationships using Claude API
"""

import json
import re
from typing import Dict, List, Any
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class GraphExtractor:
    def __init__(self, api_key: str):
        """
        Initialize graph extractor with Claude API
        
        Args:
            api_key: Anthropic API key
        """
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name="claude-3-haiku-20240307",
            temperature=0.1,
            max_tokens=1000
        )
        
        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are a knowledge graph extractor. Extract entities and relationships from the text below.

Rules:
1. Entity names should be concise (2-4 words max)
2. Entity types must be one of: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, PRODUCT
3. Relationships should be directional and meaningful
4. Common relationship types: "founded", "works_at", "located_in", "created", "part_of", "relates_to", "owns", "manages"
5. Output valid JSON only, no additional text

Text: {text}

Output JSON with this exact structure:
{{
  "entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "source entity name", "target": "target entity name", "type": "relationship_type", "description": "brief description"}}
  ]
}}

JSON Output:"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.extraction_prompt)
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with entities and relationships
        """
        if not text or not text.strip():
            return {"entities": [], "relationships": []}
        
        try:
            # Get LLM response
            response = self.chain.run(text=text[:2000])  # Limit text length
            
            # Extract JSON from response
            json_data = self._extract_json(response)
            
            # Validate and clean the data
            json_data = self._validate_extraction(json_data)
            
            return json_data
            
        except Exception as e:
            print(f"Extraction error: {str(e)}")
            # Fallback to simple extraction
            return self._fallback_extraction(text)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            Parsed JSON data
        """
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to parse the entire response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}
    
    def _validate_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted data
        
        Args:
            data: Extracted data
            
        Returns:
            Validated data
        """
        # Ensure required keys exist
        if not isinstance(data, dict):
            return {"entities": [], "relationships": []}
        
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        # Validate entities
        valid_entities = []
        entity_names = set()
        
        for entity in entities:
            if isinstance(entity, dict) and entity.get("name"):
                name = entity["name"].strip()
                if name and name not in entity_names:
                    entity_names.add(name)
                    valid_entities.append({
                        "name": name,
                        "type": entity.get("type", "CONCEPT"),
                        "description": entity.get("description", "")[:200]
                    })
        
        # Validate relationships
        valid_relationships = []
        
        for rel in relationships:
            if isinstance(rel, dict):
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                rel_type = rel.get("type", "relates_to").strip()
                
                if source and target and source != target:
                    valid_relationships.append({
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "description": rel.get("description", "")[:200]
                    })
        
        return {
            "entities": valid_entities,
            "relationships": valid_relationships
        }
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """
        Simple regex-based extraction as fallback
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with entities and relationships
        """
        entities = []
        
        # Extract capitalized words/phrases as potential entities
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Deduplicate and create entities
        seen = set()
        for match in matches:
            if match not in seen and len(match) > 2:
                seen.add(match)
                # Guess entity type based on common patterns
                entity_type = "CONCEPT"
                if any(title in match for title in ["Mr.", "Ms.", "Dr.", "Mrs."]):
                    entity_type = "PERSON"
                elif any(word in match for word in ["Inc", "Corp", "Company", "LLC"]):
                    entity_type = "ORGANIZATION"
                elif any(word in match for word in ["Street", "City", "Country", "Avenue"]):
                    entity_type = "LOCATION"
                
                entities.append({
                    "name": match,
                    "type": entity_type,
                    "description": f"Extracted from text"
                })
        
        # Limit to top 10 entities
        entities = entities[:10]
        
        # Create some basic relationships
        relationships = []
        if len(entities) > 1:
            for i in range(min(3, len(entities) - 1)):
                relationships.append({
                    "source": entities[i]["name"],
                    "target": entities[i + 1]["name"],
                    "type": "relates_to",
                    "description": "Mentioned together in text"
                })
        
        return {
            "entities": entities,
            "relationships": relationships
        }