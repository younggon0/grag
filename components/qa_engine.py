"""
Q&A Engine Module
Handles natural language questions over the knowledge graph
"""

import networkx as nx
from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class QAEngine:
    def __init__(self, api_key: str):
        """
        Initialize Q&A engine with Claude API
        
        Args:
            api_key: Anthropic API key
        """
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name="claude-3-haiku-20240307",
            temperature=0.3,
            max_tokens=500
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["graph_context", "question"],
            template="""Given this knowledge graph information, answer the question concisely.

Knowledge Graph Context:
{graph_context}

Question: {question}

Instructions:
1. Answer based ONLY on the information provided in the knowledge graph
2. If the answer is not in the graph, say "Information not found in the knowledge graph"
3. Reference specific entities and relationships when possible
4. Keep the answer concise and factual

Answer:"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
    
    def answer_question(
        self,
        question: str,
        graph: nx.DiGraph,
        entities: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """
        Answer a question based on the knowledge graph
        
        Args:
            question: User's question
            graph: NetworkX graph
            entities: List of all entities
            relationships: List of all relationships
            
        Returns:
            Answer string
        """
        try:
            # Extract relevant context from the graph
            context = self._extract_relevant_context(question, graph, entities, relationships)
            
            if not context:
                return "I couldn't find relevant information in the knowledge graph to answer your question."
            
            # Get answer from LLM
            answer = self.chain.run(
                graph_context=context,
                question=question
            )
            
            return answer.strip()
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def _extract_relevant_context(
        self,
        question: str,
        graph: nx.DiGraph,
        entities: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """
        Extract relevant context from the graph based on the question
        
        Args:
            question: User's question
            graph: NetworkX graph
            entities: List of all entities
            relationships: List of all relationships
            
        Returns:
            Context string
        """
        question_lower = question.lower()
        relevant_entities = []
        relevant_relationships = []
        
        # Find entities mentioned in the question
        mentioned_nodes = set()
        for entity in entities:
            entity_name = entity['name'].lower()
            if entity_name in question_lower or any(word in entity_name for word in question_lower.split()):
                relevant_entities.append(entity)
                mentioned_nodes.add(entity['name'])
        
        # If we found mentioned entities, get their neighborhood
        if mentioned_nodes:
            for node in mentioned_nodes:
                if graph.has_node(node):
                    # Get immediate neighbors
                    predecessors = list(graph.predecessors(node))
                    successors = list(graph.successors(node))
                    
                    # Add neighboring entities
                    for neighbor in predecessors + successors:
                        if neighbor not in mentioned_nodes:
                            if graph.has_node(neighbor):
                                node_data = graph.nodes[neighbor]
                                relevant_entities.append({
                                    'name': neighbor,
                                    'type': node_data.get('type', 'UNKNOWN'),
                                    'description': node_data.get('description', '')
                                })
                    
                    # Add relevant relationships
                    for rel in relationships:
                        if rel['source'] in mentioned_nodes or rel['target'] in mentioned_nodes:
                            relevant_relationships.append(rel)
        
        # If no specific entities found, look for question type patterns
        if not relevant_entities:
            if any(word in question_lower for word in ['who', 'person', 'people']):
                # Get all person entities
                relevant_entities = [e for e in entities if e.get('type') == 'PERSON'][:10]
            elif any(word in question_lower for word in ['where', 'location', 'place']):
                # Get all location entities
                relevant_entities = [e for e in entities if e.get('type') == 'LOCATION'][:10]
            elif any(word in question_lower for word in ['what', 'product', 'concept']):
                # Get all concept/product entities
                relevant_entities = [e for e in entities if e.get('type') in ['CONCEPT', 'PRODUCT']][:10]
            elif any(word in question_lower for word in ['when', 'event', 'time']):
                # Get all event entities
                relevant_entities = [e for e in entities if e.get('type') == 'EVENT'][:10]
            elif any(word in question_lower for word in ['company', 'organization']):
                # Get all organization entities
                relevant_entities = [e for e in entities if e.get('type') == 'ORGANIZATION'][:10]
            
            # Get relationships for these entities
            entity_names = {e['name'] for e in relevant_entities}
            relevant_relationships = [
                r for r in relationships 
                if r['source'] in entity_names or r['target'] in entity_names
            ][:20]
        
        # Build context string
        context_parts = []
        
        if relevant_entities:
            context_parts.append("Entities:")
            for entity in relevant_entities[:20]:  # Limit to prevent token overflow
                desc = f" - {entity.get('description', '')}" if entity.get('description') else ""
                context_parts.append(f"- {entity['name']} (Type: {entity['type']}){desc}")
        
        if relevant_relationships:
            context_parts.append("\nRelationships:")
            for rel in relevant_relationships[:20]:  # Limit to prevent token overflow
                desc = f" - {rel.get('description', '')}" if rel.get('description') else ""
                context_parts.append(f"- {rel['source']} --[{rel['type']}]--> {rel['target']}{desc}")
        
        # Add graph statistics if asking about general info
        if any(word in question_lower for word in ['how many', 'count', 'total', 'statistics']):
            context_parts.append(f"\nGraph Statistics:")
            context_parts.append(f"- Total entities: {len(entities)}")
            context_parts.append(f"- Total relationships: {len(relationships)}")
            
            # Count by type
            type_counts = {}
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            for entity_type, count in type_counts.items():
                context_parts.append(f"- {entity_type} entities: {count}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def get_suggested_questions(self, entities: List[Dict], relationships: List[Dict]) -> List[str]:
        """
        Generate suggested questions based on the graph content
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            List of suggested questions
        """
        suggestions = []
        
        # Get some entity names for suggestions
        person_names = [e['name'] for e in entities if e.get('type') == 'PERSON'][:3]
        org_names = [e['name'] for e in entities if e.get('type') == 'ORGANIZATION'][:3]
        
        # Person-related questions
        if person_names:
            suggestions.append(f"Who is {person_names[0]}?")
            if len(person_names) > 1:
                suggestions.append(f"What is the relationship between {person_names[0]} and {person_names[1]}?")
        
        # Organization-related questions
        if org_names:
            suggestions.append(f"What is {org_names[0]}?")
            suggestions.append(f"Who is associated with {org_names[0]}?")
        
        # General questions
        suggestions.extend([
            "How many entities are in the graph?",
            "What are the main concepts discussed?",
            "List all people mentioned in the document"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions