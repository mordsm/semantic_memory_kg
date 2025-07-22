# Conversation Memory Extractor
# Extracts semantic, episodic, and procedural memory from conversations
# Designed to work with the SemanticMemoryKG and future memory systems

import json
import re
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, Counter
import hashlib

# Try to import spacy, handle gracefully if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    SPACY_AVAILABLE = False
    spacy = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory type enumeration for future expansion
class MemoryType(Enum):
    SEMANTIC = "semantic"      # Facts, concepts, relationships
    EPISODIC = "episodic"      # Events, experiences, contexts
    PROCEDURAL = "procedural"  # Processes, procedures, how-to knowledge

@dataclass
class ConversationMessage:
    """Represents a single message in a conversation"""
    id: str
    speaker: str  # "user", "assistant", or specific person name
    content: str
    timestamp: datetime
    message_type: str = "text"  # text, code, image, etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExtractedMemory:
    """Container for extracted memory of any type"""
    memory_type: MemoryType
    content: Dict[str, Any]
    confidence: float
    source_message_id: str
    context: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if not isinstance(self.memory_type, MemoryType):
            self.memory_type = MemoryType(self.memory_type)

@dataclass
class SemanticMemoryItem:
    """Specific structure for semantic memory extraction"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    concepts: List[Dict[str, Any]]
    facts: List[Dict[str, Any]]
    definitions: List[Dict[str, Any]]

@dataclass
class EpisodicMemoryItem:
    """Structure for episodic memory (for future use)"""
    event_description: str
    participants: List[str]
    location: Optional[str]
    time_context: Dict[str, Any]
    emotional_context: Optional[str]
    outcome: Optional[str]

@dataclass
class ProceduralMemoryItem:
    """Structure for procedural memory (for future use)"""
    procedure_name: str
    steps: List[Dict[str, Any]]
    prerequisites: List[str]
    tools_required: List[str]
    success_criteria: List[str]
    common_mistakes: List[str]

class ConversationMemoryExtractor:
    """
    Extracts different types of memory from conversations.
    Designed to work with multiple memory systems (semantic, episodic, procedural).
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the conversation memory extractor
        
        Args:
            spacy_model: SpaCy model to use for NLP processing
        """
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"SpaCy model {spacy_model} not found. Install with: python -m spacy download {spacy_model}")
                logger.info("Falling back to pattern-based extraction only")
                self.nlp = None
        else:
            logger.info("SpaCy not available - using pattern-based extraction only")
            self.nlp = None
        
        # Pattern matching for different memory types
        self.semantic_patterns = self._init_semantic_patterns()
        self.episodic_patterns = self._init_episodic_patterns()
        self.procedural_patterns = self._init_procedural_patterns()
        
        # Context tracking for conversation understanding
        self.conversation_context = {
            'current_topic': None,
            'participants': set(),
            'ongoing_procedures': [],
            'mentioned_entities': defaultdict(int),
            'relationship_mentions': []
        }
    
    def _init_semantic_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for semantic memory detection"""
        return {
            'definitions': [
                r'(.+?) is (?:a|an|the) (.+?)(?:\.|$)',
                r'(.+?) (?:means|refers to|is defined as) (.+?)(?:\.|$)',
                r'(?:define|definition of) (.+?)(?:\:|$)',
                r'(.+?) can be (?:described|explained) as (.+?)(?:\.|$)'
            ],
            'facts': [
                r'(.+?) (?:has|have|contains|includes) (.+?)(?:\.|$)',
                r'(.+?) (?:was|were|is|are) (?:created|developed|invented|founded) (?:in|by|at) (.+?)(?:\.|$)',
                r'(.+?) (?:costs|weighs|measures|runs on|requires) (.+?)(?:\.|$)',
                r'the (?:capital|population|size|area) of (.+?) is (.+?)(?:\.|$)'
            ],
            'relationships': [
                r'(.+?) (?:is related to|connects to|works with|depends on) (.+?)(?:\.|$)',
                r'(.+?) and (.+?) (?:are|work) (?:together|in conjunction)(?:\.|$)',
                r'(.+?) (?:causes|leads to|results in) (.+?)(?:\.|$)',
                r'(.+?) (?:inherits from|extends|implements) (.+?)(?:\.|$)',
                r'(.+?) (?:created|developed|invented) (?:by) (.+?)(?:\.|$)',
                r'(.+?) (?:supports|includes) (.+?)(?:\.|$)'
            ],
            'concepts': [
                r'the concept of (.+?)(?:\.|$)',
                r'(.+?) is (?:a concept|an idea|a principle|a theory)(?:\.|$)',
                r'understanding (.+?) (?:involves|requires)(?:\.|$)',
                r'(.+?) (?:theory|principle|concept|paradigm)(?:\.|$)'
            ]
        }
    
    def _init_episodic_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for episodic memory detection"""
        return {
            'events': [
                r'(?:yesterday|today|last week|last month) (.+?)(?:\.|$)',
                r'(?:when|while) (?:I|we|you) (.+?)(?:\.|$)',
                r'(?:I|we|you) (?:went to|visited|attended|participated in) (.+?)(?:\.|$)',
                r'(?:during|at) (.+?) (?:I|we|you) (.+?)(?:\.|$)'
            ],
            'experiences': [
                r'(?:I|we|you) (?:experienced|felt|noticed|observed) (.+?)(?:\.|$)',
                r'(?:it was|that was) (.+?) (?:when|because) (.+?)(?:\.|$)',
                r'(?:I|we|you) (?:learned|discovered|realized) (.+?)(?:\.|$)'
            ],
            'temporal_contexts': [
                r'(?:before|after|during|while) (.+?)(?:\.|$)',
                r'(?:at|on|in) (?:\d{1,2}:\d{2}|\d{1,2}\/\d{1,2}\/\d{4}|monday|tuesday|wednesday|thursday|friday|saturday|sunday) (.+?)(?:\.|$)'
            ]
        }
    
    def _init_procedural_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for procedural memory detection"""
        return {
            'steps': [
                r'(?:first|second|third|next|then|finally|lastly) (.+?)(?:\.|$)',
                r'step \d+:? (.+?)(?:\.|$)',
                r'(?:to|in order to) (.+?), (?:you|one) (?:should|must|need to) (.+?)(?:\.|$)'
            ],
            'processes': [
                r'(?:how to|the process of|the way to) (.+?)(?:\.|$)',
                r'(?:here\'s how|this is how) (?:you|one|to) (.+?)(?:\.|$)',
                r'(?:the method|procedure|approach) for (.+?)(?:\.|$)'
            ],
            'prerequisites': [
                r'(?:before|prior to) (.+?), (?:you|one) (?:should|must|need) (.+?)(?:\.|$)',
                r'(?:requires|needs|depends on) (.+?)(?:\.|$)',
                r'(?:make sure|ensure) (?:you|one) (?:have|has) (.+?)(?:\.|$)'
            ],
            'tools_materials': [
                r'(?:using|with|via) (.+?)(?:\.|$)',
                r'(?:you will need|required tools|materials needed) (.+?)(?:\.|$)',
                r'(?:install|download|get) (.+?) (?:first|before)(?:\.|$)'
            ]
        }
    
    def extract_from_conversation(self, messages: List[ConversationMessage], 
                                 memory_types: List[MemoryType] = None) -> Dict[MemoryType, List[ExtractedMemory]]:
        """
        Extract memories from a conversation
        
        Args:
            messages: List of conversation messages
            memory_types: Types of memory to extract (default: all)
            
        Returns:
            Dictionary mapping memory types to extracted memories
        """
        if memory_types is None:
            memory_types = [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]
        
        extracted_memories = {memory_type: [] for memory_type in memory_types}
        
        # Process messages in chronological order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)
        
        for i, message in enumerate(sorted_messages):
            # Update conversation context
            self._update_context(message, sorted_messages[:i])
            
            # Extract different types of memory
            if MemoryType.SEMANTIC in memory_types:
                semantic_memories = self._extract_semantic_memory(message)
                extracted_memories[MemoryType.SEMANTIC].extend(semantic_memories)
            
            if MemoryType.EPISODIC in memory_types:
                episodic_memories = self._extract_episodic_memory(message)
                extracted_memories[MemoryType.EPISODIC].extend(episodic_memories)
            
            if MemoryType.PROCEDURAL in memory_types:
                procedural_memories = self._extract_procedural_memory(message)
                extracted_memories[MemoryType.PROCEDURAL].extend(procedural_memories)
        
        return extracted_memories
    
    def _update_context(self, current_message: ConversationMessage, 
                       previous_messages: List[ConversationMessage]):
        """Update conversation context based on current and previous messages"""
        # Debug: ensure participants is a set
        if not isinstance(self.conversation_context['participants'], set):
            self.conversation_context['participants'] = set(self.conversation_context['participants'])
            
        # Track participants
        self.conversation_context['participants'].add(current_message.speaker)
        
        # Extract mentioned entities using NLP if available
        if self.nlp:
            doc = self.nlp(current_message.content)
            for ent in doc.ents:
                if ent.text not in self.conversation_context['mentioned_entities']:
                    self.conversation_context['mentioned_entities'][ent.text] = 0
                self.conversation_context['mentioned_entities'][ent.text] += 1
        
        # Simple topic detection based on frequent terms
        words = re.findall(r'\b\w+\b', current_message.content.lower())
        word_freq = Counter(words)
        if word_freq:
            most_common = word_freq.most_common(1)[0][0]
            if len(most_common) > 3:  # Ignore short words
                self.conversation_context['current_topic'] = most_common
        
        # Ensure all context values are JSON serializable
        self.conversation_context['participants'] = list(self.conversation_context['participants'])
        self.conversation_context['mentioned_entities'] = dict(self.conversation_context['mentioned_entities'])
    
    def _extract_semantic_memory(self, message: ConversationMessage) -> List[ExtractedMemory]:
        """Extract semantic memory from a message"""
        memories = []
        content = message.content.lower()
        
        # Extract entities, relationships, concepts, facts, and definitions
        entities = self._extract_entities(message)
        relationships = self._extract_relationships(message)
        concepts = self._extract_concepts(message)
        facts = self._extract_facts(message)
        definitions = self._extract_definitions(message)
        
        # Create semantic memory item
        if any([entities, relationships, concepts, facts, definitions]):
            semantic_item = SemanticMemoryItem(
                entities=entities,
                relationships=relationships,
                concepts=concepts,
                facts=facts,
                definitions=definitions
            )
            
            memory = ExtractedMemory(
                memory_type=MemoryType.SEMANTIC,
                content=asdict(semantic_item),
                confidence=self._calculate_semantic_confidence(semantic_item),
                source_message_id=message.id,
                context={
                    'speaker': message.speaker,
                    'conversation_context': {
                        'current_topic': self.conversation_context['current_topic'],
                        'participants': list(self.conversation_context['participants']),
                        'ongoing_procedures': self.conversation_context['ongoing_procedures'],
                        'mentioned_entities': dict(self.conversation_context['mentioned_entities']),
                        'relationship_mentions': self.conversation_context['relationship_mentions']
                    },
                    'message_type': message.message_type
                },
                timestamp=message.timestamp
            )
            memories.append(memory)
        
        return memories
    
    def _extract_entities(self, message: ConversationMessage) -> List[Dict[str, Any]]:
        """Extract entities from message content"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(message.content)
            for ent in doc.ents:
                entities.append({
                    'name': ent.text,
                    'type': ent.label_,
                    'start_pos': ent.start_char,
                    'end_pos': ent.end_char,
                    'confidence': 0.8  # SpaCy entity confidence placeholder
                })
        else:
            # Fallback: Simple pattern matching for common entities
            # This is basic and should be improved with proper NLP
            patterns = {
                'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'ORG': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
                'DATE': r'\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                'TIME': r'\b\d{1,2}:\d{2}(?::\d{2})?\b'
            }
            
            for entity_type, pattern in patterns.items():
                matches = re.finditer(pattern, message.content)
                for match in matches:
                    entities.append({
                        'name': match.group(),
                        'type': entity_type,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'confidence': 0.6
                    })
        
        return entities
    
    def _extract_relationships(self, message: ConversationMessage) -> List[Dict[str, Any]]:
        """Extract relationships from message content"""
        relationships = []
        
        print(f"DEBUG: Checking message for relationships: {message.content[:100]}...")
        
        for pattern in self.semantic_patterns['relationships']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                print(f"DEBUG: Found relationship match: {match.group(0)}")
                relationships.append({
                    'source': match.group(1).strip(),
                    'target': match.group(2).strip(),
                    'relation_type': self._infer_relationship_type(match.group(0)),
                    'confidence': 0.7,
                    'context': match.group(0)
                })
        
        print(f"DEBUG: Extracted {len(relationships)} relationships from message")
        return relationships
    
    def _extract_concepts(self, message: ConversationMessage) -> List[Dict[str, Any]]:
        """Extract concepts from message content"""
        concepts = []
        
        for pattern in self.semantic_patterns['concepts']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                concepts.append({
                    'name': match.group(1).strip(),
                    'category': 'general_concept',
                    'definition': self._extract_concept_definition(match.group(0)),
                    'confidence': 0.6,
                    'context': match.group(0)
                })
        
        return concepts
    
    def _extract_facts(self, message: ConversationMessage) -> List[Dict[str, Any]]:
        """Extract facts from message content"""
        facts = []
        
        for pattern in self.semantic_patterns['facts']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                facts.append({
                    'subject': match.group(1).strip(),
                    'predicate': self._extract_predicate(match.group(0)),
                    'object': match.group(2).strip() if len(match.groups()) > 1 else None,
                    'confidence': 0.8,
                    'context': match.group(0)
                })
        
        return facts
    
    def _extract_definitions(self, message: ConversationMessage) -> List[Dict[str, Any]]:
        """Extract definitions from message content"""
        definitions = []
        
        for pattern in self.semantic_patterns['definitions']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                definitions.append({
                    'term': match.group(1).strip(),
                    'definition': match.group(2).strip(),
                    'confidence': 0.9,
                    'context': match.group(0)
                })
        
        return definitions
    
    def _extract_episodic_memory(self, message: ConversationMessage) -> List[ExtractedMemory]:
        """Extract episodic memory from a message (placeholder for future implementation)"""
        # This is a placeholder for episodic memory extraction
        # Full implementation would analyze temporal contexts, events, and experiences
        memories = []
        
        # Basic event detection
        events = []
        for pattern in self.episodic_patterns['events']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                events.append({
                    'event_description': match.group(1).strip(),
                    'temporal_context': self._extract_temporal_context(match.group(0)),
                    'participants': [message.speaker],
                    'confidence': 0.6
                })
        
        if events:
            episodic_item = EpisodicMemoryItem(
                event_description=events[0]['event_description'] if events else "",
                participants=list(self.conversation_context['participants']),
                location=None,  # Would be extracted with better NLP
                time_context={'message_timestamp': message.timestamp.isoformat()},
                emotional_context=None,  # Would be extracted with sentiment analysis
                outcome=None
            )
            
            memory = ExtractedMemory(
                memory_type=MemoryType.EPISODIC,
                content=asdict(episodic_item),
                confidence=0.6,
                source_message_id=message.id,
                context={
                    'speaker': message.speaker,
                    'conversation_context': {
                        'current_topic': self.conversation_context['current_topic'],
                        'participants': list(self.conversation_context['participants']),
                        'ongoing_procedures': self.conversation_context['ongoing_procedures'],
                        'mentioned_entities': dict(self.conversation_context['mentioned_entities']),
                        'relationship_mentions': self.conversation_context['relationship_mentions']
                    }
                },
                timestamp=message.timestamp
            )
            memories.append(memory)
        
        return memories
    
    def _extract_procedural_memory(self, message: ConversationMessage) -> List[ExtractedMemory]:
        """Extract procedural memory from a message (placeholder for future implementation)"""
        memories = []
        
        # Detect procedural content
        steps = []
        for pattern in self.procedural_patterns['steps']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                steps.append({
                    'step_description': match.group(1).strip(),
                    'order': len(steps) + 1,
                    'confidence': 0.7
                })
        
        # Detect processes
        processes = []
        for pattern in self.procedural_patterns['processes']:
            matches = re.finditer(pattern, message.content, re.IGNORECASE)
            for match in matches:
                processes.append(match.group(1).strip())
        
        if steps or processes:
            procedural_item = ProceduralMemoryItem(
                procedure_name=processes[0] if processes else "unnamed_procedure",
                steps=[{'description': step['step_description'], 'order': step['order']} for step in steps],
                prerequisites=[],  # Would be extracted with better analysis
                tools_required=[],  # Would be extracted from tool patterns
                success_criteria=[],
                common_mistakes=[]
            )
            
            memory = ExtractedMemory(
                memory_type=MemoryType.PROCEDURAL,
                content=asdict(procedural_item),
                confidence=self._calculate_procedural_confidence(procedural_item),
                source_message_id=message.id,
                context={
                    'speaker': message.speaker,
                    'conversation_context': {
                        'current_topic': self.conversation_context['current_topic'],
                        'participants': list(self.conversation_context['participants']),
                        'ongoing_procedures': self.conversation_context['ongoing_procedures'],
                        'mentioned_entities': dict(self.conversation_context['mentioned_entities']),
                        'relationship_mentions': self.conversation_context['relationship_mentions']
                    }
                },
                timestamp=message.timestamp
            )
            memories.append(memory)
        
        return memories
    
    # Helper methods
    def _infer_relationship_type(self, context: str) -> str:
        """Infer relationship type from context"""
        context_lower = context.lower()
        if any(word in context_lower for word in ['related', 'connects', 'linked']):
            return 'related_to'
        elif any(word in context_lower for word in ['causes', 'leads to', 'results']):
            return 'causes'
        elif any(word in context_lower for word in ['depends', 'requires']):
            return 'depends_on'
        elif any(word in context_lower for word in ['inherits', 'extends', 'implements']):
            return 'inherits_from'
        else:
            return 'general_relation'
    
    def _extract_concept_definition(self, context: str) -> str:
        """Extract concept definition from context"""
        # Simple extraction - would be improved with better NLP
        return context.strip()
    
    def _extract_predicate(self, context: str) -> str:
        """Extract predicate from fact context"""
        # Simple predicate extraction
        predicates = ['has', 'is', 'contains', 'includes', 'requires', 'costs']
        for predicate in predicates:
            if predicate in context.lower():
                return predicate
        return 'relates_to'
    
    def _extract_temporal_context(self, context: str) -> Dict[str, Any]:
        """Extract temporal context from text"""
        temporal_indicators = ['yesterday', 'today', 'last week', 'last month', 'recently']
        for indicator in temporal_indicators:
            if indicator in context.lower():
                return {'temporal_indicator': indicator}
        return {}
    
    def _calculate_semantic_confidence(self, semantic_item: SemanticMemoryItem) -> float:
        """Calculate confidence score for semantic memory"""
        total_items = (len(semantic_item.entities) + len(semantic_item.relationships) + 
                      len(semantic_item.concepts) + len(semantic_item.facts) + 
                      len(semantic_item.definitions))
        
        if total_items == 0:
            return 0.0
        
        # Weight different types of semantic content
        weights = {'entities': 0.8, 'relationships': 0.9, 'concepts': 0.7, 
                  'facts': 0.9, 'definitions': 1.0}
        
        weighted_score = (
            len(semantic_item.entities) * weights['entities'] +
            len(semantic_item.relationships) * weights['relationships'] +
            len(semantic_item.concepts) * weights['concepts'] +
            len(semantic_item.facts) * weights['facts'] +
            len(semantic_item.definitions) * weights['definitions']
        ) / total_items
        
        return min(weighted_score, 1.0)
    
    def _calculate_procedural_confidence(self, procedural_item: ProceduralMemoryItem) -> float:
        """Calculate confidence score for procedural memory"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on structure
        if len(procedural_item.steps) > 1:
            confidence += 0.3
        if procedural_item.prerequisites:
            confidence += 0.1
        if procedural_item.tools_required:
            confidence += 0.1
        
        return min(confidence, 1.0)


class MemoryIntegrationService:
    """
    Service for integrating extracted memories with different memory systems
    """
    
    def __init__(self, semantic_kg=None):
        """
        Initialize with memory systems
        
        Args:
            semantic_kg: Instance of SemanticMemoryKG
        """
        self.semantic_kg = semantic_kg
        # Placeholders for future memory systems
        self.episodic_memory = None  # Will be implemented later
        self.procedural_memory = None  # Will be implemented later
        
        logger.info("MemoryIntegrationService initialized")
    
    def integrate_extracted_memories(self, extracted_memories: Dict[MemoryType, List[ExtractedMemory]]) -> Dict[str, Any]:
        """
        Integrate extracted memories into appropriate memory systems
        
        Args:
            extracted_memories: Dictionary of extracted memories by type
            
        Returns:
            Integration results and statistics
        """
        results = {
            'semantic': {'added': 0, 'errors': 0},
            'episodic': {'added': 0, 'errors': 0},
            'procedural': {'added': 0, 'errors': 0}
        }
        
        # Integrate semantic memories
        if MemoryType.SEMANTIC in extracted_memories and self.semantic_kg:
            results['semantic'] = self._integrate_semantic_memories(
                extracted_memories[MemoryType.SEMANTIC]
            )
        
        # Placeholders for future integration
        if MemoryType.EPISODIC in extracted_memories:
            results['episodic'] = self._integrate_episodic_memories(
                extracted_memories[MemoryType.EPISODIC]
            )
        
        if MemoryType.PROCEDURAL in extracted_memories:
            results['procedural'] = self._integrate_procedural_memories(
                extracted_memories[MemoryType.PROCEDURAL]
            )
        
        return results
    
    def _integrate_semantic_memories(self, semantic_memories: List[ExtractedMemory]) -> Dict[str, int]:
        """Integrate semantic memories into the knowledge graph"""
        added = 0
        errors = 0
        
        for memory in semantic_memories:
            try:
                semantic_content = memory.content
                
                # Add entities
                for entity_data in semantic_content.get('entities', []):
                    entity_id = self.semantic_kg.add_entity(
                        name=entity_data['name'],
                        entity_type=entity_data['type'],
                        properties={
                            'confidence': entity_data.get('confidence', 0.5),
                            'source': memory.source_message_id,
                            'extracted_at': memory.timestamp.isoformat()
                        }
                    )
                    added += 1
                
                # Add relationships
                for rel_data in semantic_content.get('relationships', []):
                    try:
                        # First ensure both entities exist
                        source_entity = self.semantic_kg.add_entity(
                            name=rel_data['source'],
                            entity_type='extracted_entity',
                            properties={'extracted_from': memory.source_message_id}
                        )
                        target_entity = self.semantic_kg.add_entity(
                            name=rel_data['target'],
                            entity_type='extracted_entity', 
                            properties={'extracted_from': memory.source_message_id}
                        )
                        
                        # Then create the relationship
                        self.semantic_kg.add_relationship(
                            source_name=rel_data['source'],
                            target_name=rel_data['target'],
                            relation_type=rel_data['relation_type'],
                            properties={
                                'confidence': rel_data.get('confidence', 0.5),
                                'source': memory.source_message_id,
                                'context': rel_data.get('context', '')
                            },
                            confidence=rel_data.get('confidence', 0.5)
                        )
                        added += 1
                        logger.info(f"Added relationship: {rel_data['source']} -> {rel_data['target']}")
                    except ValueError as e:
                        logger.warning(f"Could not add relationship: {e}")
                        errors += 1
                    except Exception as e:
                        logger.error(f"Error creating relationship: {e}")
                        errors += 1
                
                # Add concepts
                for concept_data in semantic_content.get('concepts', []):
                    concept_id = self.semantic_kg.add_concept(
                        name=concept_data['name'],
                        definition=concept_data.get('definition', ''),
                        category=concept_data.get('category', 'general'),
                        properties={
                            'confidence': concept_data.get('confidence', 0.5),
                            'source': memory.source_message_id,
                            'context': concept_data.get('context', '')
                        }
                    )
                    added += 1
                
                # Add facts as entities with special properties
                for fact_data in semantic_content.get('facts', []):
                    fact_id = self.semantic_kg.add_entity(
                        name=f"Fact: {fact_data['subject']} {fact_data['predicate']}",
                        entity_type='fact',
                        properties={
                            'subject': fact_data['subject'],
                            'predicate': fact_data['predicate'],
                            'object': fact_data.get('object'),
                            'confidence': fact_data.get('confidence', 0.5),
                            'source': memory.source_message_id,
                            'context': fact_data.get('context', '')
                        }
                    )
                    added += 1
                
                # Add definitions as concepts
                for def_data in semantic_content.get('definitions', []):
                    def_id = self.semantic_kg.add_concept(
                        name=def_data['term'],
                        definition=def_data['definition'],
                        category='definition',
                        properties={
                            'confidence': def_data.get('confidence', 0.5),
                            'source': memory.source_message_id,
                            'context': def_data.get('context', '')
                        }
                    )
                    added += 1
                
            except Exception as e:
                logger.error(f"Error integrating semantic memory: {e}")
                errors += 1
        
        return {'added': added, 'errors': errors}
    
    def _integrate_episodic_memories(self, episodic_memories: List[ExtractedMemory]) -> Dict[str, int]:
        """Placeholder for episodic memory integration"""
        # This will be implemented when episodic memory system is created
        logger.info(f"Would integrate {len(episodic_memories)} episodic memories")
        return {'added': 0, 'errors': 0}
    
    def _integrate_procedural_memories(self, procedural_memories: List[ExtractedMemory]) -> Dict[str, int]:
        """Placeholder for procedural memory integration"""
        # This will be implemented when procedural memory system is created
        logger.info(f"Would integrate {len(procedural_memories)} procedural memories")
        return {'added': 0, 'errors': 0}


# Example usage
if __name__ == "__main__":
    # Example conversation
    messages = [
        ConversationMessage(
            id="msg_1",
            speaker="user",
            content="What is Python? I heard it's a programming language.",
            timestamp=datetime.now() - timedelta(minutes=10)
        ),
        ConversationMessage(
            id="msg_2",
            speaker="assistant",
            content="Python is a high-level programming language. It was created by Guido van Rossum in 1991. Python supports multiple programming paradigms including procedural and object-oriented programming.",
            timestamp=datetime.now() - timedelta(minutes=9)
        ),
        ConversationMessage(
            id="msg_3",
            speaker="user",
            content="How do I install Python on my computer? What are the steps?",
            timestamp=datetime.now() - timedelta(minutes=8)
        ),
        ConversationMessage(
            id="msg_4",
            speaker="assistant",
            content="To install Python, first go to python.org. Then download the latest version for your operating system. Next, run the installer and make sure to check 'Add Python to PATH'. Finally, verify the installation by opening terminal and typing 'python --version'.",
            timestamp=datetime.now() - timedelta(minutes=7)
        ),
        ConversationMessage(
            id="msg_5",
            speaker="user",
            content="Yesterday I attended a Python conference in San Francisco. I learned about Django framework there.",
            timestamp=datetime.now() - timedelta(minutes=6)
        )
    ]
    
    # Initialize extractor
    extractor = ConversationMemoryExtractor()
    
    # Extract memories from conversation
    extracted_memories = extractor.extract_from_conversation(messages)
    
    # Display results
    print("=== EXTRACTED MEMORIES ===")
    for memory_type, memories in extracted_memories.items():
        print(f"\n{memory_type.value.upper()} MEMORIES: {len(memories)}")
        for i, memory in enumerate(memories):
            print(f"\n  Memory {i+1}:")
            print(f"    Confidence: {memory.confidence:.2f}")
            print(f"    Source: {memory.source_message_id}")
            print(f"    Speaker: {memory.context.get('speaker')}")
            
            if memory.memory_type == MemoryType.SEMANTIC:
                content = memory.content
                if content.get('entities'):
                    print(f"    Entities: {[e['name'] for e in content['entities']]}")
                if content.get('relationships'):
                    print(f"    Relationships: {len(content['relationships'])}")
                if content.get('concepts'):
                    print(f"    Concepts: {[c['name'] for c in content['concepts']]}")
                if content.get('facts'):
                    print(f"    Facts: {len(content['facts'])}")
                if content.get('definitions'):
                    print(f"    Definitions: {[d['term'] for d in content['definitions']]}")
            
            elif memory.memory_type == MemoryType.PROCEDURAL:
                content = memory.content
                print(f"    Procedure: {content.get('procedure_name')}")
                print(f"    Steps: {len(content.get('steps', []))}")
            
            elif memory.memory_type == MemoryType.EPISODIC:
                content = memory.content
                print(f"    Event: {content.get('event_description')}")
                print(f"    Participants: {content.get('participants')}")
    
    print("\n=== INTEGRATION WITH SEMANTIC MEMORY ===")
    
    # Example integration with SemanticMemoryKG (assuming it's available)
    try:
        from semantic_memory_kg import SemanticMemoryKG  # Import your KG
        
        # Initialize knowledge graph
        kg = SemanticMemoryKG(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        # Initialize integration service
        integration_service = MemoryIntegrationService(semantic_kg=kg)
        
        # Integrate extracted memories
        integration_results = integration_service.integrate_extracted_memories(extracted_memories)
        
        print("Integration Results:")
        for memory_type, results in integration_results.items():
            print(f"  {memory_type}: Added {results['added']}, Errors {results['errors']}")
        
        # Query the knowledge graph to see what was added
        print("\n=== KNOWLEDGE GRAPH CONTENTS ===")
        stats = kg.get_statistics()
        print(f"Total entities: {stats['entities']['total']}")
        print(f"Total relationships: {stats['relationships']['total']}")
        print(f"Total concepts: {stats['concepts']['total']}")
        
        # Show some entities
        entities = kg.query_entities(entity_type="programming_language")
        print(f"\nProgramming languages found: {[e.name for e in entities]}")
        
        concepts = kg.query_concepts(category="definition")
        print(f"Definitions found: {[c.name for c in concepts]}")
        
        kg.close()
        
    except ImportError:
        print("SemanticMemoryKG not available - would integrate memories here")
        print("Integration Results (simulated):")
        for memory_type, memories in extracted_memories.items():
            print(f"  {memory_type.value}: Would integrate {len(memories)} memories")


# Advanced Usage Examples and Utilities

class ConversationAnalyzer:
    """Advanced analyzer for conversation patterns and memory density"""
    
    def __init__(self, extractor: ConversationMemoryExtractor):
        self.extractor = extractor
    
    def analyze_conversation_quality(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """Analyze the quality and density of extractable memories in a conversation"""
        extracted_memories = self.extractor.extract_from_conversation(messages)
        
        total_messages = len(messages)
        total_memories = sum(len(memories) for memories in extracted_memories.values())
        
        # Calculate memory density per message
        memory_density = total_memories / total_messages if total_messages > 0 else 0
        
        # Analyze memory type distribution
        memory_distribution = {
            memory_type.value: len(memories) 
            for memory_type, memories in extracted_memories.items()
        }
        
        # Calculate average confidence per memory type
        confidence_by_type = {}
        for memory_type, memories in extracted_memories.items():
            if memories:
                avg_confidence = sum(m.confidence for m in memories) / len(memories)
                confidence_by_type[memory_type.value] = avg_confidence
            else:
                confidence_by_type[memory_type.value] = 0.0
        
        # Identify information-rich messages
        message_memory_counts = []
        for msg in messages:
            msg_memories = self.extractor.extract_from_conversation([msg])
            msg_count = sum(len(memories) for memories in msg_memories.values())
            message_memory_counts.append({
                'message_id': msg.id,
                'speaker': msg.speaker,
                'memory_count': msg_count,
                'content_length': len(msg.content)
            })
        
        # Sort by memory richness
        message_memory_counts.sort(key=lambda x: x['memory_count'], reverse=True)
        
        return {
            'total_messages': total_messages,
            'total_memories': total_memories,
            'memory_density': memory_density,
            'memory_distribution': memory_distribution,
            'confidence_by_type': confidence_by_type,
            'richest_messages': message_memory_counts[:5],  # Top 5 memory-rich messages
            'quality_score': self._calculate_quality_score(
                memory_density, confidence_by_type, memory_distribution
            )
        }
    
    def _calculate_quality_score(self, memory_density: float, 
                                confidence_by_type: Dict[str, float],
                                memory_distribution: Dict[str, int]) -> float:
        """Calculate overall conversation quality score for memory extraction"""
        # Base score from memory density
        density_score = min(memory_density * 0.5, 1.0)  # Cap at 1.0
        
        # Confidence score (average across all types)
        avg_confidence = sum(confidence_by_type.values()) / len(confidence_by_type) if confidence_by_type else 0
        confidence_score = avg_confidence
        
        # Diversity score (how many different memory types are present)
        present_types = sum(1 for count in memory_distribution.values() if count > 0)
        diversity_score = present_types / 3.0  # 3 total memory types
        
        # Combined score
        quality_score = (density_score * 0.4 + confidence_score * 0.4 + diversity_score * 0.2)
        
        return quality_score


class MemoryExportService:
    """Service for exporting extracted memories in various formats"""
    
    @staticmethod
    def _serialize_for_json(obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, MemoryType):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return obj
    
    @staticmethod
    def _deep_serialize(obj):
        """Recursively serialize nested objects"""
        if isinstance(obj, dict):
            return {key: MemoryExportService._deep_serialize(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [MemoryExportService._deep_serialize(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, MemoryType):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return MemoryExportService._deep_serialize(obj.__dict__)
        else:
            return obj
    
    @staticmethod
    def export_to_json(extracted_memories: Dict[MemoryType, List[ExtractedMemory]], 
                      filepath: str) -> str:
        """Export extracted memories to JSON format"""
        export_data = {}
        
        for memory_type, memories in extracted_memories.items():
            export_data[memory_type.value] = []
            for memory in memories:
                memory_data = {
                    'memory_type': memory.memory_type.value,
                    'content': MemoryExportService._deep_serialize(memory.content),
                    'confidence': memory.confidence,
                    'source_message_id': memory.source_message_id,
                    'context': MemoryExportService._deep_serialize(memory.context),
                    'timestamp': memory.timestamp.isoformat()
                }
                export_data[memory_type.value].append(memory_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=MemoryExportService._serialize_for_json)
        
        logger.info(f"Exported memories to {filepath}")
        return filepath
    
    @staticmethod
    def export_semantic_to_kg_format(semantic_memories: List[ExtractedMemory], 
                                   filepath: str) -> str:
        """Export semantic memories in a format ready for knowledge graph import"""
        entities = []
        relationships = []
        concepts = []
        
        for memory in semantic_memories:
            content = memory.content
            source_info = {
                'source_message': memory.source_message_id,
                'confidence': memory.confidence,
                'extracted_at': memory.timestamp.isoformat()
            }
            
            # Collect entities
            for entity in content.get('entities', []):
                entities.append({
                    **entity,
                    **source_info
                })
            
            # Collect relationships
            for relationship in content.get('relationships', []):
                relationships.append({
                    **relationship,
                    **source_info
                })
            
            # Collect concepts
            for concept in content.get('concepts', []):
                concepts.append({
                    **concept,
                    **source_info
                })
        
        kg_export_data = {
            'entities': entities,
            'relationships': relationships,
            'concepts': concepts,
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_memories_processed': len(semantic_memories)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(kg_export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported semantic memories in KG format to {filepath}")
        return filepath


class ConversationMemoryBatch:
    """Utility for batch processing multiple conversations"""
    
    def __init__(self, extractor: ConversationMemoryExtractor, 
                 integration_service: MemoryIntegrationService = None):
        self.extractor = extractor
        self.integration_service = integration_service
        self.analyzer = ConversationAnalyzer(extractor)
    
    def process_conversation_batch(self, conversations: List[List[ConversationMessage]], 
                                 integrate: bool = True) -> Dict[str, Any]:
        """Process multiple conversations and extract memories"""
        batch_results = {
            'processed_conversations': 0,
            'total_memories': {'semantic': 0, 'episodic': 0, 'procedural': 0},
            'integration_results': {'semantic': {'added': 0, 'errors': 0}},
            'quality_scores': [],
            'processing_errors': []
        }
        
        for i, conversation in enumerate(conversations):
            try:
                # Extract memories
                extracted_memories = self.extractor.extract_from_conversation(conversation)
                
                # Update totals
                for memory_type, memories in extracted_memories.items():
                    batch_results['total_memories'][memory_type.value] += len(memories)
                
                # Integrate if service available
                if integrate and self.integration_service:
                    integration_results = self.integration_service.integrate_extracted_memories(extracted_memories)
                    
                    # Update integration results
                    for memory_type, results in integration_results.items():
                        if memory_type not in batch_results['integration_results']:
                            batch_results['integration_results'][memory_type] = {'added': 0, 'errors': 0}
                        batch_results['integration_results'][memory_type]['added'] += results['added']
                        batch_results['integration_results'][memory_type]['errors'] += results['errors']
                
                # Analyze conversation quality
                quality_analysis = self.analyzer.analyze_conversation_quality(conversation)
                batch_results['quality_scores'].append({
                    'conversation_index': i,
                    'quality_score': quality_analysis['quality_score'],
                    'memory_density': quality_analysis['memory_density'],
                    'total_memories': quality_analysis['total_memories']
                })
                
                batch_results['processed_conversations'] += 1
                
            except Exception as e:
                logger.error(f"Error processing conversation {i}: {e}")
                batch_results['processing_errors'].append({
                    'conversation_index': i,
                    'error': str(e)
                })
        
        # Calculate average quality score
        if batch_results['quality_scores']:
            avg_quality = sum(score['quality_score'] for score in batch_results['quality_scores']) / len(batch_results['quality_scores'])
            batch_results['average_quality_score'] = avg_quality
        else:
            batch_results['average_quality_score'] = 0.0
        
        return batch_results


# Configuration and Settings
class MemoryExtractionConfig:
    """Configuration class for memory extraction settings"""
    
    def __init__(self):
        # NLP Settings
        self.spacy_model = "en_core_web_sm"
        self.confidence_threshold = 0.5
        
        # Memory Type Settings
        self.extract_semantic = True
        self.extract_episodic = True
        self.extract_procedural = True
        
        # Pattern Matching Settings
        self.case_sensitive = False
        self.max_pattern_matches = 10
        
        # Integration Settings
        self.auto_integrate = True
        self.batch_size = 100
        
        # Export Settings
        self.auto_export = False
        self.export_format = "json"  # json, csv, xml
        self.export_path = "exports/"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'spacy_model': self.spacy_model,
            'confidence_threshold': self.confidence_threshold,
            'extract_semantic': self.extract_semantic,
            'extract_episodic': self.extract_episodic,
            'extract_procedural': self.extract_procedural,
            'case_sensitive': self.case_sensitive,
            'max_pattern_matches': self.max_pattern_matches,
            'auto_integrate': self.auto_integrate,
            'batch_size': self.batch_size,
            'auto_export': self.auto_export,
            'export_format': self.export_format,
            'export_path': self.export_path
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryExtractionConfig':
        """Create configuration from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# Example of complete workflow
def example_complete_workflow():
    """Example showing complete memory extraction and integration workflow"""
    
    # Sample conversation data
    sample_messages = [
        ConversationMessage(
            id="msg_1",
            speaker="user", 
            content="I'm learning about machine learning. What is neural network?",
            timestamp=datetime.now() - timedelta(hours=2)
        ),
        ConversationMessage(
            id="msg_2",
            speaker="assistant",
            content="A neural network is a computational model inspired by biological neural networks. It consists of interconnected nodes called neurons that process information. Neural networks are used for pattern recognition and machine learning tasks.",
            timestamp=datetime.now() - timedelta(hours=2, minutes=-5)
        ),
        ConversationMessage(
            id="msg_3",
            speaker="user",
            content="How do I train a neural network? What are the steps?",
            timestamp=datetime.now() - timedelta(hours=1, minutes=50)
        ),
        ConversationMessage(
            id="msg_4",
            speaker="assistant", 
            content="To train a neural network: First, prepare your dataset by cleaning and normalizing the data. Second, initialize the network weights randomly. Third, forward propagate through the network. Fourth, calculate the loss using a loss function. Fifth, backpropagate to compute gradients. Finally, update the weights using an optimizer like SGD or Adam.",
            timestamp=datetime.now() - timedelta(hours=1, minutes=45)
        )
    ]
    
    print("=== COMPLETE MEMORY EXTRACTION WORKFLOW ===")
    
    # Initialize components
    config = MemoryExtractionConfig()
    extractor = ConversationMemoryExtractor(config.spacy_model)
    
    # Extract memories
    extracted_memories = extractor.extract_from_conversation(sample_messages)
    
    # Analyze conversation
    analyzer = ConversationAnalyzer(extractor)
    analysis = analyzer.analyze_conversation_quality(sample_messages)
    
    print(f"Conversation Quality Score: {analysis['quality_score']:.2f}")
    print(f"Memory Density: {analysis['memory_density']:.2f} memories/message")
    print(f"Total Memories Extracted: {analysis['total_memories']}")
    
    # Export memories
    export_service = MemoryExportService()
    json_file = export_service.export_to_json(extracted_memories, "extracted_memories.json")
    print(f"Memories exported to: {json_file}")
    
    # Export semantic memories in KG format
    if MemoryType.SEMANTIC in extracted_memories:
        kg_file = export_service.export_semantic_to_kg_format(
            extracted_memories[MemoryType.SEMANTIC], 
            "semantic_memories_kg.json"
        )
        print(f"Semantic memories for KG exported to: {kg_file}")


if __name__ == "__main__":
    example_complete_workflow()