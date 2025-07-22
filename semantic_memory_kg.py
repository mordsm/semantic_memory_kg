# Semantic Memory Knowledge Graph with Neo4j
# A self-sustained system for storing and retrieving semantic knowledge

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import re

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
except ImportError:
    raise ImportError("Please install neo4j driver: pip install neo4j")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a semantic entity in the knowledge graph"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Concept:
    """Represents an abstract concept in semantic memory"""
    id: str
    name: str
    definition: str
    category: str
    related_entities: List[str]
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class SemanticMemoryKG:
    """
    Neo4j-based knowledge graph for semantic memory storage and retrieval.
    Can work independently or as part of a larger LangGraph system.
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password",
                 database: str = "neo4j"):
        """
        Initialize the semantic memory knowledge graph with Neo4j
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._init_constraints()
            logger.info("Connected to Neo4j successfully")
        except ServiceUnavailable:
            logger.error("Failed to connect to Neo4j. Make sure Neo4j is running.")
            raise
    
    def _init_constraints(self):
        """Initialize constraints and indices in Neo4j"""
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
            
            # Create indices for better performance
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            session.run("CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)")
            session.run("CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category)")
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID based on content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None,
                   embedding: List[float] = None) -> str:
        """Add a new entity to the knowledge graph"""
        if properties is None:
            properties = {}
        
        entity_id = self._generate_id(f"{name}_{entity_type}")
        
        with self.driver.session(database=self.database) as session:
            # Check if entity exists
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e",
                id=entity_id
            )
            if result.single():
                logger.info(f"Entity {name} already exists")
                return entity_id
            
            # Create entity
            query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                type: $type,
                properties: $properties,
                embedding: $embedding,
                created_at: datetime(),
                updated_at: datetime()
            })
            RETURN e.id as id
            """
            
            result = session.run(query, 
                id=entity_id,
                name=name,
                type=entity_type,
                properties=json.dumps(properties),
                embedding=embedding
            )
            
            logger.info(f"Added entity: {name} ({entity_type})")
            return entity_id
    
    def add_relationship(self, source_name: str, target_name: str, relation_type: str,
                        properties: Dict[str, Any] = None, confidence: float = 1.0) -> str:
        """Add a relationship between entities"""
        if properties is None:
            properties = {}
        
        with self.driver.session(database=self.database) as session:
            # Find source and target entities
            query = """
            MATCH (source:Entity {name: $source_name})
            MATCH (target:Entity {name: $target_name})
            MERGE (source)-[r:RELATES {
                type: $relation_type,
                properties: $properties,
                confidence: $confidence,
                created_at: datetime(),
                updated_at: datetime()
            }]->(target)
            RETURN r
            """
            
            result = session.run(query,
                source_name=source_name,
                target_name=target_name,
                relation_type=relation_type,
                properties=json.dumps(properties),
                confidence=confidence
            )
            
            if result.single():
                logger.info(f"Added relationship: {source_name} --{relation_type}--> {target_name}")
                return self._generate_id(f"{source_name}_{relation_type}_{target_name}")
            else:
                raise ValueError(f"Could not find entities: {source_name}, {target_name}")
    
    def add_concept(self, name: str, definition: str, category: str,
                   related_entities: List[str] = None, properties: Dict[str, Any] = None,
                   embedding: List[float] = None) -> str:
        """Add a new concept to semantic memory"""
        if related_entities is None:
            related_entities = []
        if properties is None:
            properties = {}
        
        concept_id = self._generate_id(f"{name}_{category}")
        
        with self.driver.session(database=self.database) as session:
            # Create concept
            query = """
            CREATE (c:Concept {
                id: $id,
                name: $name,
                definition: $definition,
                category: $category,
                related_entities: $related_entities,
                properties: $properties,
                embedding: $embedding,
                created_at: datetime(),
                updated_at: datetime()
            })
            RETURN c.id as id
            """
            
            result = session.run(query,
                id=concept_id,
                name=name,
                definition=definition,
                category=category,
                related_entities=related_entities,
                properties=json.dumps(properties),
                embedding=embedding
            )
            
            # Link to related entities
            for entity_name in related_entities:
                session.run("""
                    MATCH (c:Concept {id: $concept_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (c)-[:RELATES_TO]->(e)
                """, concept_id=concept_id, entity_name=entity_name)
            
            logger.info(f"Added concept: {name} ({category})")
            return concept_id
    
    def query_entities(self, entity_type: str = None, name_pattern: str = None,
                      properties_filter: Dict[str, Any] = None) -> List[Entity]:
        """Query entities based on various criteria"""
        with self.driver.session(database=self.database) as session:
            query = "MATCH (e:Entity) WHERE 1=1"
            params = {}
            
            if entity_type:
                query += " AND e.type = $entity_type"
                params['entity_type'] = entity_type
            
            if name_pattern:
                query += " AND e.name =~ $name_pattern"
                params['name_pattern'] = f"(?i).*{name_pattern}.*"
            
            query += " RETURN e"
            
            result = session.run(query, **params)
            
            entities = []
            for record in result:
                node = record['e']
                entity = Entity(
                    id=node['id'],
                    name=node['name'],
                    type=node['type'],
                    properties=json.loads(node.get('properties', '{}')),
                    embedding=node.get('embedding'),
                    created_at=node.get('created_at'),
                    updated_at=node.get('updated_at')
                )
                entities.append(entity)
            
            return entities
    
    def query_relationships(self, relation_type: str = None, source_name: str = None,
                           target_name: str = None) -> List[Relationship]:
        """Query relationships based on various criteria"""
        with self.driver.session(database=self.database) as session:
            query = "MATCH (source:Entity)-[r:RELATES]->(target:Entity) WHERE 1=1"
            params = {}
            
            if relation_type:
                query += " AND r.type = $relation_type"
                params['relation_type'] = relation_type
            
            if source_name:
                query += " AND source.name = $source_name"
                params['source_name'] = source_name
            
            if target_name:
                query += " AND target.name = $target_name"
                params['target_name'] = target_name
            
            query += " RETURN r, source.id as source_id, target.id as target_id"
            
            result = session.run(query, **params)
            
            relationships = []
            for record in result:
                rel = record['r']
                relationship = Relationship(
                    id=self._generate_id(f"{record['source_id']}_{rel['type']}_{record['target_id']}"),
                    source_id=record['source_id'],
                    target_id=record['target_id'],
                    relation_type=rel['type'],
                    properties=json.loads(rel.get('properties', '{}')),
                    confidence=rel.get('confidence', 1.0),
                    created_at=rel.get('created_at'),
                    updated_at=rel.get('updated_at')
                )
                relationships.append(relationship)
            
            return relationships
    
    def query_concepts(self, category: str = None, name_pattern: str = None) -> List[Concept]:
        """Query concepts based on various criteria"""
        with self.driver.session(database=self.database) as session:
            query = "MATCH (c:Concept) WHERE 1=1"
            params = {}
            
            if category:
                query += " AND c.category = $category"
                params['category'] = category
            
            if name_pattern:
                query += " AND c.name =~ $name_pattern"
                params['name_pattern'] = f"(?i).*{name_pattern}.*"
            
            query += " RETURN c"
            
            result = session.run(query, **params)
            
            concepts = []
            for record in result:
                node = record['c']
                concept = Concept(
                    id=node['id'],
                    name=node['name'],
                    definition=node['definition'],
                    category=node['category'],
                    related_entities=node.get('related_entities', []),
                    properties=json.loads(node.get('properties', '{}')),
                    embedding=node.get('embedding'),
                    created_at=node.get('created_at'),
                    updated_at=node.get('updated_at')
                )
                concepts.append(concept)
            
            return concepts
    
    def get_entity_neighborhood(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth"""
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (center:Entity {{name: $entity_name}})
            OPTIONAL MATCH (center)-[r:RELATES*1..{depth}]-(neighbor:Entity)
            RETURN center, collect(DISTINCT neighbor) as neighbors, 
                   collect(DISTINCT r) as relationships
            """
            
            result = session.run(query, entity_name=entity_name)
            record = result.single()
            
            if not record:
                return {}
            
            center_node = record['center']
            center_entity = Entity(
                id=center_node['id'],
                name=center_node['name'],
                type=center_node['type'],
                properties=json.loads(center_node.get('properties', '{}')),
                embedding=center_node.get('embedding'),
                created_at=center_node.get('created_at'),
                updated_at=center_node.get('updated_at')
            )
            
            neighbors = []
            for neighbor_node in record['neighbors']:
                if neighbor_node:  # Check if not None
                    neighbor = Entity(
                        id=neighbor_node['id'],
                        name=neighbor_node['name'],
                        type=neighbor_node['type'],
                        properties=json.loads(neighbor_node.get('properties', '{}')),
                        embedding=neighbor_node.get('embedding'),
                        created_at=neighbor_node.get('created_at'),
                        updated_at=neighbor_node.get('updated_at')
                    )
                    neighbors.append(neighbor)
            
            return {
                'center_entity': center_entity,
                'neighbors': neighbors,
                'relationships': record['relationships']
            }
    
    def find_path(self, source_name: str, target_name: str, max_length: int = 5) -> List[str]:
        """Find shortest path between two entities"""
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (source:Entity {{name: $source_name}})
            MATCH (target:Entity {{name: $target_name}})
            MATCH path = shortestPath((source)-[*1..{max_length}]-(target))
            RETURN [node in nodes(path) | node.name] as path
            """
            
            result = session.run(query, source_name=source_name, target_name=target_name)
            record = result.single()
            
            return record['path'] if record else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        with self.driver.session(database=self.database) as session:
            # Count entities by type
            entity_stats = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
            """)
            
            # Count all relationships (any type)
            all_rels = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
            """)
            
            # Count concepts by category
            concept_stats = session.run("""
                MATCH (c:Concept)
                RETURN c.category as category, count(c) as count
            """)
            
            # Overall counts
            total_stats = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH ()-[r]->()
                OPTIONAL MATCH (c:Concept)
                RETURN count(DISTINCT e) as entities, 
                       count(DISTINCT r) as relationships, 
                       count(DISTINCT c) as concepts
            """)
            
            total_record = total_stats.single()
            
            # Debug: Show what relationships actually exist
            debug_rels = session.run("MATCH ()-[r]->() RETURN type(r), count(r)")
            
            stats = {
                'entities': {
                    'total': total_record['entities'],
                    'by_type': {record['type']: record['count'] for record in entity_stats}
                },
                'relationships': {
                    'total': total_record['relationships'],
                    'by_type': {record['rel_type']: record['count'] for record in all_rels}
                },
                'concepts': {
                    'total': total_record['concepts'],
                    'by_category': {record['category']: record['count'] for record in concept_stats}
                },
                'debug_relationships': list(debug_rels)
            }
            
            return stats
    
    def delete_entity(self, entity_name: str):
        """Delete an entity and its relationships"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Entity {name: $entity_name})
            DETACH DELETE e
            """
            session.run(query, entity_name=entity_name)
            logger.info(f"Deleted entity: {entity_name}")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        logger.info("Closed Neo4j connection")


# Integration interface for LangGraph
class SemanticMemoryInterface:
    """Interface for integrating SemanticMemoryKG with LangGraph systems"""
    
    def __init__(self, semantic_kg: SemanticMemoryKG):
        self.semantic_kg = semantic_kg
    
    def add_knowledge_from_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """Extract and add knowledge from text"""
        result = {'entities_added': [], 'relationships_added': [], 'concepts_added': []}
        
        # Add source as document entity
        doc_id = self.semantic_kg.add_entity(
            name=f"Document_{source}",
            entity_type="document",
            properties={'source': source, 'content': text, 'extracted_from_text': True}
        )
        result['entities_added'].append(doc_id)
        
        return result
    
    def query_semantic_memory(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the semantic memory with natural language"""
        results = {
            'entities': self.semantic_kg.query_entities(name_pattern=query),
            'relationships': [],
            'concepts': self.semantic_kg.query_concepts(name_pattern=query)
        }
        
        return results
    
    def get_context_for_entity(self, entity_name: str) -> Dict[str, Any]:
        """Get rich context for an entity"""
        neighborhood = self.semantic_kg.get_entity_neighborhood(entity_name, depth=2)
        
        context = {
            'entity': neighborhood.get('center_entity'),
            'related_entities': neighborhood.get('neighbors', []),
            'relationships': neighborhood.get('relationships', []),
            'concepts': []
        }
        
        return context


# Example usage
if __name__ == "__main__":
    # Initialize with Neo4j
    kg = SemanticMemoryKG(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    # Add sample data
    kg.add_entity("Python", "programming_language", 
                  {"paradigm": "multi-paradigm", "year": 1991})
    kg.add_entity("Java", "programming_language", 
                  {"paradigm": "object-oriented", "year": 1995})
    
    # Add relationship
    kg.add_relationship("Python", "Java", "similar_to", 
                       {"similarity": 0.8})
    
    # Add concept
    kg.add_concept("Object-Oriented Programming", 
                   "Programming paradigm based on objects",
                   "programming_concept", 
                   related_entities=["Python", "Java"])
    
    # Query examples
    entities = kg.query_entities(entity_type="programming_language")
    print(f"Found {len(entities)} programming languages")
    
    # Get statistics
    stats = kg.get_statistics()
    print(f"Total entities: {stats['entities']['total']}")
    print(f"Total relationships: {stats['relationships']['total']}")
    print(f"Total concepts: {stats['concepts']['total']}")
    
    kg.close()