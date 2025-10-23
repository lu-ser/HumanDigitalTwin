"""
LangGraph pipeline per costruire il Knowledge Graph da triplette validate.

Workflow:
1. Prende triplette (opzionalmente validate con ontology)
2. Genera broad_topic e narrow_topic per ogni tripletta (LLM)
3. Verifica se i topic esistono già nel KG
4. Salva tripletta + metadata nel Knowledge Graph
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


# ==================== PYDANTIC MODELS ====================

class TopicClassification(BaseModel):
    """Classificazione di una tripletta in broad e narrow topics."""
    broader_topic: str = Field(description="Categoria ampia (es. 'Health', 'Finance', 'Social')")
    narrower_topic: str = Field(description="Sottocategoria specifica (es. 'Heart Rate', 'Expenses', 'Friends')")
    reasoning: str = Field(description="Breve spiegazione della classificazione")


class TopicMatchResult(BaseModel):
    """Risultato del matching semantico di un topic con una lista esistente."""
    match_found: bool = Field(description="True se è stato trovato un topic simile nella lista, False altrimenti. MUST be a boolean (true/false), not a string.")
    matched_topic: str = Field(default="", description="Il topic esistente che matcha (vuoto se match_found=False)")
    reasoning: str = Field(description="Spiegazione del match o del mancato match")

    @field_validator('match_found', mode='before')
    @classmethod
    def validate_bool(cls, v):
        """Converti stringhe in boolean se necessario."""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes')
        return v


# ==================== STATE DEFINITION ====================

class KGBuilderState(TypedDict):
    """
    State per il Knowledge Graph Builder.

    Campi:
    - triplets: Lista di triplette da processare
    - ontology_check_enabled: Flag per abilitare/disabilitare check ontologico
    - current_triplet_index: Indice della tripletta corrente
    - current_processed_triplet: Tripletta corrente in processing (sovrascritta ad ogni iterazione)
    - processed_triplets: Lista finale di triplette processate (accumulata)
    - existing_topics: Set di coppie (broader, narrower) già esistenti nel KG
    - errors: Lista di errori durante il processing
    - matched_broader_topic: Broader topic matchato dall'LLM (None se nuovo)
    - matched_narrower_topic: Narrower topic matchato dall'LLM (None se nuovo)
    """
    triplets: List[Dict[str, Any]]
    ontology_check_enabled: bool
    current_triplet_index: int
    current_processed_triplet: Dict[str, Any]  # Triplet corrente (non accumula)
    processed_triplets: List[Dict[str, Any]]  # Lista finale (gestita manualmente)
    existing_topics: Dict[str, List[str]]  # {broader_topic: [narrower_topic1, narrower_topic2, ...]}
    errors: List[str]  # Lista errori (gestita manualmente)
    matched_broader_topic: Optional[str]  # Broader topic matchato (None se nuovo)
    matched_narrower_topic: Optional[str]  # Narrower topic matchato (None se nuovo)


# ==================== NEO4J STORAGE (with LangChain Integration) ====================

class Neo4jKnowledgeGraph:
    """
    Storage Neo4j per il Knowledge Graph con integrazione LangChain.

    Struttura del grafo:
    - (:Person {id, name}) - Root node per ogni profilo
    - Nodi entità estratti dalle triplette (con label dal type LLM)
    - Relazioni tra entità (dal predicate), con proprietà:
      - broader_topic: categoria ampia
      - narrower_topic: sottocategoria
      - reasoning: spiegazione classificazione

    Esempio:
    (:Person {name: "Mario"})-[:BELONGS_TO]->(:Person {id: "mario_rossi"})
    (:Person {id: "mario_rossi"})-[goesForAWalk {broader_topic: "Hobbies", narrower_topic: "Walking"}]->(:Place {name: "Parco"})
    """

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j", person_id: str = "main_person", person_name: str = "User"):
        """
        Inizializza la connessione a Neo4j.

        Args:
            uri: URI del database Neo4j (es. "bolt://localhost:7687")
            username: Username per autenticazione
            password: Password per autenticazione
            database: Nome del database (default: "neo4j")
            person_id: ID univoco della Person (default: "main_person")
            person_name: Nome della Person (default: "User")
        """
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.person_id = person_id
        self.person_name = person_name

        # Crea root node e constraints se non esistono
        self._setup_schema()

    def _setup_schema(self):
        """Crea constraints e root node del KG."""
        with self.driver.session(database=self.database) as session:
            # Rimuovi vecchi constraints globali se esistono (legacy)
            try:
                session.run("DROP CONSTRAINT broader_topic_name IF EXISTS")
                session.run("DROP CONSTRAINT narrower_topic_name IF EXISTS")
                session.run("DROP CONSTRAINT broader_topic_composite IF EXISTS")
                session.run("DROP CONSTRAINT narrower_topic_composite IF EXISTS")
            except Exception:
                pass  # Constraints potrebbero non esistere

            # Constraint per Person.id
            session.run("CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")

            # Crea root node Person (ontologia Schema.org)
            session.run("""
                MERGE (person:Person {id: $person_id})
                ON CREATE SET person.created_at = datetime(), person.name = $person_name
                ON MATCH SET person.last_accessed = datetime()
            """, person_id=self.person_id, person_name=self.person_name)

    def close(self):
        """Chiude la connessione al database."""
        if self.driver:
            self.driver.close()

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """
        Ritorna tutti i profili Person nel database.

        Returns:
            Lista di dict con info su ogni Person
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Person)
                RETURN p.id AS id, p.name AS name, p.created_at AS created_at, p.last_accessed AS last_accessed
                ORDER BY p.created_at DESC
            """)
            return [dict(record) for record in result]

    def delete_person(self, person_id: str) -> bool:
        """
        Cancella un profilo Person e tutto il suo grafo.

        Args:
            person_id: ID della Person da cancellare

        Returns:
            True se cancellato, False altrimenti
        """
        with self.driver.session(database=self.database) as session:
            # Cancella Person e tutti i nodi collegati (cascade)
            result = session.run("""
                MATCH (p:Person {id: $person_id})
                OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(b:BroaderTopic)-[:HAS_SUBCATEGORY]->(n:NarrowerTopic)-[:CONTAINS]->(t:Triplet)
                DETACH DELETE p, b, n, t
                RETURN count(p) AS deleted
            """, person_id=person_id)
            record = result.single()
            return record["deleted"] > 0 if record else False

    def cleanup_legacy_nodes(self) -> Dict[str, int]:
        """
        Rimuove nodi BroaderTopic, NarrowerTopic e Triplet del vecchio schema.
        Utile dopo migration a schema entità-relazioni.

        Returns:
            Dict con numero di nodi rimossi per tipo
        """
        with self.driver.session(database=self.database) as session:
            # Rimuovi tutti i nodi del vecchio schema
            result = session.run("""
                MATCH (b:BroaderTopic)
                OPTIONAL MATCH (b)-[:HAS_SUBCATEGORY]->(n:NarrowerTopic)
                OPTIONAL MATCH (n)-[:CONTAINS]->(t:Triplet)
                DETACH DELETE b, n, t
                RETURN count(DISTINCT b) AS deleted_broader,
                       count(DISTINCT n) AS deleted_narrower,
                       count(DISTINCT t) AS deleted_triplets
            """)
            record = result.single()

            return {
                "deleted_broader_topics": record["deleted_broader"] if record else 0,
                "deleted_narrower_topics": record["deleted_narrower"] if record else 0,
                "deleted_triplets": record["deleted_triplets"] if record else 0
            }

    def get_all_broader_topics(self) -> List[str]:
        """
        Ritorna tutti i broader topics esistenti nel grafo per questa Person.
        Estrae i topic dalle proprietà delle relazioni.

        Returns:
            Lista di nomi dei broader topics (unici)
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (person:Person {id: $person_id})-[:KNOWS]->(n)-[r]->(m)
                WHERE r.person_id = $person_id AND r.broader_topic IS NOT NULL
                RETURN DISTINCT r.broader_topic AS name
                ORDER BY name
            """, person_id=self.person_id)
            return [record["name"] for record in result]

    def get_narrower_topics_for_broader(self, broader_topic: str) -> List[str]:
        """
        Ritorna tutti i narrower topics per un dato broader topic di questa Person.
        Estrae i topic dalle proprietà delle relazioni.

        Args:
            broader_topic: Broad topic

        Returns:
            Lista di nomi dei narrower topics (unici)
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (person:Person {id: $person_id})-[:KNOWS]->(n)-[r]->(m)
                WHERE r.person_id = $person_id
                  AND r.broader_topic = $broader
                  AND r.narrower_topic IS NOT NULL
                RETURN DISTINCT r.narrower_topic AS name
                ORDER BY name
            """, person_id=self.person_id, broader=broader_topic)
            return [record["name"] for record in result]

    def topic_exists(self, broader_topic: str, narrower_topic: str = None) -> bool:
        """
        Verifica se un topic esiste già nel grafo di questa Person.
        Cerca nelle proprietà delle relazioni.

        Args:
            broader_topic: Broad topic da verificare
            narrower_topic: Narrow topic da verificare (opzionale)

        Returns:
            True se il topic esiste, False altrimenti
        """
        with self.driver.session(database=self.database) as session:
            if narrower_topic is None:
                # Check solo broader
                result = session.run("""
                    MATCH (person:Person {id: $person_id})-[:KNOWS]->()-[r]->()
                    WHERE r.person_id = $person_id AND r.broader_topic = $broader
                    RETURN count(r) > 0 AS exists
                """, person_id=self.person_id, broader=broader_topic)
            else:
                # Check entrambi
                result = session.run("""
                    MATCH (person:Person {id: $person_id})-[:KNOWS]->()-[r]->()
                    WHERE r.person_id = $person_id
                      AND r.broader_topic = $broader
                      AND r.narrower_topic = $narrower
                    RETURN count(r) > 0 AS exists
                """, person_id=self.person_id, broader=broader_topic, narrower=narrower_topic)

            record = result.single()
            return record["exists"] if record else False

    def add_triplet(self, triplet: Dict[str, Any], broader_topic: str, narrower_topic: str) -> None:
        """
        Aggiunge una tripletta al grafo creando nodi entità e relazioni.

        Args:
            triplet: Tripletta da aggiungere (con subject, predicate, object e i loro tipi)
            broader_topic: Broad topic (usato come proprietà della relazione)
            narrower_topic: Narrow topic (usato come proprietà della relazione)
        """
        # Estrai valori e tipi dalla tripletta
        def get_value(field):
            val = triplet.get(field, "")
            if isinstance(val, dict):
                return val.get("value", str(val))
            return str(val)

        def get_type(field):
            val = triplet.get(field, {})
            if isinstance(val, dict):
                return val.get("type", "Entity")  # Default type se non specificato
            return "Entity"

        subject_value = get_value("subject")
        predicate_value = get_value("predicate")
        object_value = get_value("object")

        subject_type = get_type("subject")
        predicate_type = get_type("predicate")  # Non usato per ora, ma disponibile
        object_type = get_type("object")

        # Normalizza il predicate per usarlo come nome relazione (CamelCase, no spazi)
        import re
        predicate_rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', predicate_value)
        predicate_rel_type = ''.join(word.capitalize() for word in predicate_rel_type.split('_'))

        # Metadata per la relazione
        reasoning = triplet.get("classification_reasoning", "")

        with self.driver.session(database=self.database) as session:
            # Query Cypher dinamica (il tipo di relazione non può essere parametrizzato)
            # NOTA: Uso 'obj' invece di 'object' perché è una keyword riservata in Neo4j
            query = f"""
                // Match o crea il nodo Subject
                MERGE (subj:{subject_type} {{name: $subject_value, person_id: $person_id}})
                ON CREATE SET subj.created_at = datetime()

                // Match o crea il nodo Object
                MERGE (obj:{object_type} {{name: $object_value, person_id: $person_id}})
                ON CREATE SET obj.created_at = datetime()

                // Crea la relazione con i topic come proprietà
                MERGE (subj)-[r:{predicate_rel_type}]->(obj)
                ON CREATE SET
                    r.broader_topic = $broader_topic,
                    r.narrower_topic = $narrower_topic,
                    r.reasoning = $reasoning,
                    r.predicate_original = $predicate_original,
                    r.created_at = datetime(),
                    r.person_id = $person_id
                ON MATCH SET
                    r.last_used = datetime()

                // Link subject alla Person se non esiste già
                WITH subj, obj, r
                MATCH (person:Person {{id: $person_id}})
                MERGE (person)-[:KNOWS]->(subj)

                RETURN subj, obj, r
            """

            session.run(query,
                person_id=self.person_id,
                subject_value=subject_value,
                object_value=object_value,
                broader_topic=broader_topic,
                narrower_topic=narrower_topic,
                reasoning=reasoning,
                predicate_original=predicate_value
            )

        logger.info(f"Added triplet to Neo4j KG: ({subject_value})-[{predicate_rel_type}]->({object_value}) [{broader_topic}/{narrower_topic}]")

    def get_all_topics(self) -> Dict[str, List[str]]:
        """
        Ritorna tutti i topic nel grafo di questa Person.
        Estrae dalle proprietà delle relazioni.

        Returns:
            Dict con {broader_topic: [narrower_topic1, narrower_topic2, ...]}
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (person:Person {id: $person_id})-[:KNOWS]->()-[r]->()
                WHERE r.person_id = $person_id
                  AND r.broader_topic IS NOT NULL
                  AND r.narrower_topic IS NOT NULL
                RETURN r.broader_topic AS broader, collect(DISTINCT r.narrower_topic) AS narrowers
                ORDER BY broader
            """, person_id=self.person_id)

            topics_dict = {}
            for record in result:
                broader = record["broader"]
                if broader not in topics_dict:
                    topics_dict[broader] = []
                topics_dict[broader].extend(record["narrowers"])

            # Rimuovi duplicati
            return {k: sorted(list(set(v))) for k, v in topics_dict.items()}

    def get_stats(self) -> Dict[str, int]:
        """
        Ritorna statistiche sul grafo di questa Person.

        Returns:
            Dict con statistiche (num_broader_topics, num_narrower_topics, num_relationships)
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (person:Person {id: $person_id})-[:KNOWS]->()-[r]->()
                WHERE r.person_id = $person_id
                RETURN
                    count(DISTINCT r.broader_topic) AS num_broader,
                    count(DISTINCT r.narrower_topic) AS num_narrower,
                    count(r) AS num_relationships
            """, person_id=self.person_id)

            record = result.single()
            return {
                "num_broader_topics": record["num_broader"] if record else 0,
                "num_narrower_topics": record["num_narrower"] if record else 0,
                "num_triplets": record["num_relationships"] if record else 0  # Keep same key for compatibility
            }

    def to_plotly_network(self, max_triplets_per_topic: int = 5):
        """
        Genera una visualizzazione network interattiva con Plotly del grafo entità-relazioni.

        Args:
            max_triplets_per_topic: Numero massimo di relazioni da mostrare (non usato, mantiene compatibilità)

        Returns:
            Figure Plotly
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx
        except ImportError:
            logger.error("plotly or networkx not installed")
            return None

        # Recupera entità e relazioni da Neo4j per questa Person
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (person:Person {id: $person_id})-[:KNOWS]->(subject)-[r]->(object)
                WHERE r.person_id = $person_id
                RETURN
                    labels(subject)[0] AS subject_type,
                    subject.name AS subject_name,
                    type(r) AS rel_type,
                    r.broader_topic AS broader_topic,
                    r.narrower_topic AS narrower_topic,
                    labels(object)[0] AS object_type,
                    object.name AS object_name
                LIMIT 100
            """, person_id=self.person_id)

            # Costruisci grafo NetworkX
            G = nx.DiGraph()
            G.add_node('Person', label=self.person_name, node_type='person', size=30)

            # Mappa entità per evitare duplicati
            entity_nodes = {}

            for record in result:
                subject_name = record["subject_name"]
                subject_type = record["subject_type"]
                object_name = record["object_name"]
                object_type = record["object_type"]
                rel_type = record["rel_type"]
                broader = record["broader_topic"]
                narrower = record["narrower_topic"]

                # Crea nodo subject se non esiste
                subject_id = f"{subject_type}_{subject_name}"
                if subject_id not in entity_nodes:
                    G.add_node(subject_id, label=f"{subject_name}", node_type=subject_type, size=20)
                    entity_nodes[subject_id] = True
                    # Link alla Person
                    G.add_edge('Person', subject_id, label='KNOWS')

                # Crea nodo object se non esiste
                object_id = f"{object_type}_{object_name}"
                if object_id not in entity_nodes:
                    G.add_node(object_id, label=f"{object_name}", node_type=object_type, size=20)
                    entity_nodes[object_id] = True

                # Crea relazione con label
                edge_label = f"{rel_type}\n[{narrower}]"
                G.add_edge(subject_id, object_id, label=edge_label, rel_type=rel_type, broader=broader, narrower=narrower)

        # Layout grafo
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Traces per archi con label
        edge_traces = []
        edge_annotations = []

        for edge in G.edges(data=True):
            source, target, data = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # Linea dell'arco
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)

            # Annotazione per la label (al centro dell'arco)
            edge_label = data.get('label', '')
            if edge_label and edge_label != 'KNOWS':  # Non mostrare KNOWS
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2

                annotation = dict(
                    x=mid_x,
                    y=mid_y,
                    text=edge_label,
                    showarrow=False,
                    font=dict(size=9, color='#555'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    borderpad=2
                )
                edge_annotations.append(annotation)

        # Raccolta tipi entità unici per generazione colori dinamica
        entity_types = set()
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'Entity')
            if node_type != 'person':  # Escludi il nodo root Person
                entity_types.add(node_type)

        # Genera color map dinamica
        import colorsys
        color_map = {'person': '#4A90E2'}  # Person root sempre blu

        # Genera colori per ogni tipo di entità
        num_types = len(entity_types)
        for i, entity_type in enumerate(sorted(entity_types)):
            # Usa HSV per distribuire colori uniformemente
            hue = i / max(num_types, 1)
            saturation = 0.7
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            color_map[entity_type] = hex_color

        # Traces per nodi
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = G.nodes[node]
            label = node_data.get('label', node)
            node_text.append(label)
            node_type = node_data.get('node_type', 'Entity')
            node_color.append(color_map.get(node_type, '#888888'))
            node_size.append(node_data.get('size', 15))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=11, family='Arial'),
            hoverinfo='text',
            marker=dict(color=node_color, size=node_size, line=dict(width=2, color='white')),
            showlegend=False
        )

        # Combina tutti i traces (archi + nodi)
        all_traces = edge_traces + [node_trace]

        fig = go.Figure(data=all_traces,
                       layout=go.Layout(
                           title=dict(text='Knowledge Graph Structure (Neo4j)', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=700,
                           annotations=edge_annotations  # Aggiungi le annotazioni delle label
                       ))

        return fig

    def get_entity_types_legend(self) -> Dict[str, str]:
        """
        Ritorna i tipi di entità presenti nel grafo con i loro colori.

        Returns:
            Dict con {entity_type: hex_color}
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (person:Person {id: $person_id})-[:KNOWS]->(n)
                WHERE n.person_id = $person_id
                RETURN DISTINCT labels(n)[0] AS entity_type
                ORDER BY entity_type
            """, person_id=self.person_id)

            entity_types = [record["entity_type"] for record in result]

            # Genera gli stessi colori del grafo
            import colorsys
            color_map = {}

            num_types = len(entity_types)
            for i, entity_type in enumerate(sorted(entity_types)):
                hue = i / max(num_types, 1)
                saturation = 0.7
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                color_map[entity_type] = hex_color

            return color_map


# ==================== IN-MEMORY STORAGE (Fallback) ====================

class InMemoryKnowledgeGraph:
    """
    Storage temporaneo in-memory per il Knowledge Graph.

    IMPORTANTE: Questa classe è un placeholder temporaneo.
    Sarà sostituita con Neo4j nelle prossime iterazioni.
    """

    def __init__(self):
        # Structure: {broader_topic: {narrower_topic: [triplets]}}
        self.graph: Dict[str, Dict[str, List[Dict]]] = {}

    def get_all_broader_topics(self) -> List[str]:
        """
        Ritorna tutti i broader topics esistenti nel grafo.

        Returns:
            Lista di nomi dei broader topics
        """
        return sorted(list(self.graph.keys()))

    def get_narrower_topics_for_broader(self, broader_topic: str) -> List[str]:
        """
        Ritorna tutti i narrower topics per un dato broader topic.

        Args:
            broader_topic: Broad topic

        Returns:
            Lista di nomi dei narrower topics
        """
        if broader_topic not in self.graph:
            return []
        return sorted(list(self.graph[broader_topic].keys()))

    def topic_exists(self, broader_topic: str, narrower_topic: str = None) -> bool:
        """
        Verifica se un topic esiste già nel grafo.

        Args:
            broader_topic: Broad topic da verificare
            narrower_topic: Narrow topic da verificare (opzionale)

        Returns:
            True se il topic esiste, False altrimenti
        """
        if broader_topic not in self.graph:
            return False

        if narrower_topic is None:
            return True

        return narrower_topic in self.graph[broader_topic]

    def add_triplet(self, triplet: Dict[str, Any], broader_topic: str, narrower_topic: str) -> None:
        """
        Aggiunge una tripletta al grafo con i suoi topic.

        Args:
            triplet: Tripletta da aggiungere
            broader_topic: Broad topic
            narrower_topic: Narrow topic
        """
        # Crea broader_topic se non esiste
        if broader_topic not in self.graph:
            self.graph[broader_topic] = {}

        # Crea narrower_topic se non esiste
        if narrower_topic not in self.graph[broader_topic]:
            self.graph[broader_topic][narrower_topic] = []

        # Aggiungi tripletta
        self.graph[broader_topic][narrower_topic].append(triplet)

        logger.info(f"Added triplet to KG: {broader_topic} → {narrower_topic}")

    def get_all_topics(self) -> Dict[str, List[str]]:
        """
        Ritorna tutti i topic nel grafo.

        Returns:
            Dict con {broader_topic: [narrower_topic1, narrower_topic2, ...]}
        """
        return {
            broader: list(self.graph[broader].keys())
            for broader in self.graph
        }

    def get_stats(self) -> Dict[str, int]:
        """
        Ritorna statistiche sul grafo.

        Returns:
            Dict con statistiche (num_broader_topics, num_narrower_topics, num_triplets)
        """
        num_broader = len(self.graph)
        num_narrower = sum(len(narrowers) for narrowers in self.graph.values())
        num_triplets = sum(
            len(triplets)
            for narrowers in self.graph.values()
            for triplets in narrowers.values()
        )

        return {
            "num_broader_topics": num_broader,
            "num_narrower_topics": num_narrower,
            "num_triplets": num_triplets
        }

    def to_plotly_network(self, max_triplets_per_topic: int = 5):
        """
        Genera una visualizzazione network interattiva con Plotly (NO Graphviz richiesto).

        Args:
            max_triplets_per_topic: Numero massimo di triplette da mostrare per topic

        Returns:
            Figure Plotly
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx
        except ImportError:
            logger.error("plotly or networkx not installed")
            return None

        # Crea grafo NetworkX
        G = nx.DiGraph()

        # Root node
        G.add_node('KG', label='Knowledge Graph', node_type='root')

        node_id = 0

        # Aggiungi nodi e archi
        for broader_topic, narrower_topics in self.graph.items():
            broader_id = f'broader_{node_id}'
            node_id += 1

            G.add_node(broader_id, label=broader_topic, node_type='broader')
            G.add_edge('KG', broader_id)

            for narrower_topic, triplets in narrower_topics.items():
                narrower_id = f'narrower_{node_id}'
                node_id += 1

                G.add_node(narrower_id, label=narrower_topic, node_type='narrower')
                G.add_edge(broader_id, narrower_id)

                # Mostra solo prime N triplette
                displayed_triplets = triplets[:max_triplets_per_topic]

                for idx, triplet in enumerate(displayed_triplets):
                    triplet_id = f'triplet_{node_id}'
                    node_id += 1

                    # Estrai valori
                    def get_value(field):
                        val = triplet.get(field, "")
                        if isinstance(val, dict):
                            return val.get("value", str(val))
                        return str(val)

                    subject = get_value("subject")
                    predicate = get_value("predicate")
                    obj = get_value("object")

                    label = f"{subject[:20]}...\n→{predicate[:15]}...\n→{obj[:20]}..." if len(subject) > 20 else f"{subject}\n→{predicate}\n→{obj}"

                    G.add_node(triplet_id, label=label, node_type='triplet')
                    G.add_edge(narrower_id, triplet_id)

                # Aggiungi nodo "more" se necessario
                if len(triplets) > max_triplets_per_topic:
                    more_id = f'more_{node_id}'
                    node_id += 1
                    G.add_node(more_id, label=f'+{len(triplets) - max_triplets_per_topic} more', node_type='more')
                    G.add_edge(narrower_id, more_id)

        # Layout gerarchico
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Estrai coordinate
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Nodi
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        color_map = {
            'root': '#4A90E2',
            'broader': '#7ED321',
            'narrower': '#F5A623',
            'triplet': '#E8E8E8',
            'more': '#D3D3D3'
        }

        size_map = {
            'root': 30,
            'broader': 25,
            'narrower': 20,
            'triplet': 12,
            'more': 10
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_data = G.nodes[node]
            node_text.append(node_data.get('label', node))
            node_type = node_data.get('node_type', 'triplet')
            node_color.append(color_map.get(node_type, '#888'))
            node_size.append(size_map.get(node_type, 15))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=10),
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            )
        )

        # Crea figura
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text='Knowledge Graph Structure', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))

        return fig

    def to_graphviz(self, format: str = 'svg', max_triplets_per_topic: int = 5) -> str:
        """
        Genera una visualizzazione Graphviz del Knowledge Graph.

        Args:
            format: Formato di output ('svg', 'png', 'pdf', 'dot')
            max_triplets_per_topic: Numero massimo di triplette da mostrare per topic (default: 5)

        Returns:
            String con il grafo in formato richiesto (SVG, PNG, PDF) o DOT source
        """
        try:
            from graphviz import Digraph
        except ImportError:
            logger.error("graphviz not installed. Run: pip install graphviz")
            return ""

        # Crea il grafo
        dot = Digraph(comment='Knowledge Graph', format=format)
        dot.attr(rankdir='LR')  # Left to Right layout
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr('edge', fontname='Arial', fontsize='10')

        # Root node
        dot.node('KG', 'Knowledge Graph', fillcolor='#4A90E2', fontcolor='white', shape='ellipse')

        # Itera sui broader topics
        for broader_topic, narrower_topics in self.graph.items():
            broader_id = f'broader_{broader_topic}'
            dot.node(broader_id, broader_topic, fillcolor='#7ED321', fontcolor='white')
            dot.edge('KG', broader_id, label='category')

            # Itera sui narrower topics
            for narrower_topic, triplets in narrower_topics.items():
                narrower_id = f'narrower_{broader_topic}_{narrower_topic}'
                dot.node(narrower_id, narrower_topic, fillcolor='#F5A623', fontcolor='white')
                dot.edge(broader_id, narrower_id, label='subcategory')

                # Mostra solo le prime N triplette per evitare grafici troppo grandi
                displayed_triplets = triplets[:max_triplets_per_topic]
                num_hidden = len(triplets) - len(displayed_triplets)

                for idx, triplet in enumerate(displayed_triplets):
                    # Estrai valori dalla tripletta
                    def get_value(field):
                        val = triplet.get(field, "")
                        if isinstance(val, dict):
                            return val.get("value", str(val))
                        return str(val)

                    subject = get_value("subject")
                    predicate = get_value("predicate")
                    obj = get_value("object")

                    # Crea nodo per la tripletta
                    triplet_id = f'triplet_{broader_topic}_{narrower_topic}_{idx}'
                    triplet_label = f'{subject}\\n→ {predicate} →\\n{obj}'
                    dot.node(triplet_id, triplet_label, fillcolor='#E8E8E8', fontcolor='black',
                            fontsize='9', shape='note')
                    dot.edge(narrower_id, triplet_id, style='dashed', color='gray')

                # Se ci sono triplette nascoste, aggiungi un nodo informativo
                if num_hidden > 0:
                    hidden_id = f'hidden_{broader_topic}_{narrower_topic}'
                    dot.node(hidden_id, f'... +{num_hidden} more', fillcolor='lightgray',
                            fontcolor='gray', fontsize='8', shape='plaintext')
                    dot.edge(narrower_id, hidden_id, style='dotted', color='lightgray')

        # Ritorna il risultato in base al formato
        if format == 'dot':
            return dot.source
        else:
            # Renderizza e ritorna i dati binari o il path
            return dot.pipe(format=format).decode('utf-8') if format == 'svg' else dot.pipe(format=format)


# ==================== KNOWLEDGE GRAPH BUILDER ====================

class KnowledgeGraphBuilder:
    """
    LangGraph pipeline per costruire il Knowledge Graph.
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_model: str = "llama-3.3-70b-versatile",
        prompt_manager=None,
        storage: InMemoryKnowledgeGraph = None,
        enable_logging: bool = True
    ):
        """
        Inizializza il builder.

        Args:
            llm_api_key: API key per Groq
            llm_model: Modello LLM da usare
            prompt_manager: Istanza di PromptManager
            storage: Storage per il KG (default: InMemoryKnowledgeGraph)
            enable_logging: Abilita logging Rich con AgentLogger
        """
        # LLM
        self.llm = ChatGroq(
            groq_api_key=llm_api_key,
            model_name=llm_model,
            temperature=0.3,
            max_tokens=2000
        )

        # Prompt Manager
        self.prompt_manager = prompt_manager
        if not self.prompt_manager:
            from src.prompts import PromptManager
            self.prompt_manager = PromptManager()

        # Storage
        self.storage = storage or InMemoryKnowledgeGraph()

        # Logger
        self.enable_logging = enable_logging
        self.logger = None
        if enable_logging:
            from src.utils import get_logger
            self.logger = get_logger()

        # Grafo
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Costruisce il grafo LangGraph con matching semantico LLM."""
        workflow = StateGraph(KGBuilderState)

        # Nodi
        workflow.add_node("generate_topics", self._generate_topics_node)
        workflow.add_node("match_broader_topic", self._match_broader_topic_node)
        workflow.add_node("match_narrower_topic", self._match_narrower_topic_node)
        workflow.add_node("store_triplet", self._store_triplet_node)

        # Entry point
        workflow.set_entry_point("generate_topics")

        # Conditional da generate_topics: se triplet vuoto skippa, altrimenti continua
        workflow.add_conditional_edges(
            "generate_topics",
            self._check_triplet_valid,
            {
                "valid": "match_broader_topic",  # Matcha il broader topic
                "skip": "generate_topics",  # Retry con prossima tripletta
                "end": END  # Fine se non ci sono più triplette
            }
        )

        # Da match_broader_topic -> match_narrower_topic
        workflow.add_edge("match_broader_topic", "match_narrower_topic")

        # Da match_narrower_topic -> store_triplet
        workflow.add_edge("match_narrower_topic", "store_triplet")

        # Conditional: continua con prossima tripletta o termina
        workflow.add_conditional_edges(
            "store_triplet",
            self._should_continue,
            {
                "continue": "generate_topics",
                "end": END
            }
        )

        return workflow.compile(
            checkpointer=None,
            interrupt_before=None,
            interrupt_after=None,
            debug=False
        )

    # ==================== NODES ====================

    def _generate_topics_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Genera broad_topic e narrow_topic per la tripletta corrente usando LLM.
        """
        idx = state["current_triplet_index"]
        triplets = state["triplets"]

        if idx >= len(triplets):
            return state

        triplet = triplets[idx]

        # Estrai valori dalla tripletta (gestisce sia formato {value: ..., type: ...} che formato semplice stringa)
        def extract_value(field):
            """Estrae il valore da un campo che può essere dict o stringa."""
            try:
                if not isinstance(triplet, dict):
                    logger.error(f"Triplet is not a dict, it's {type(triplet)}: {triplet}")
                    return ""

                value = triplet.get(field, "")
                if isinstance(value, dict):
                    return value.get("value", "")
                elif isinstance(value, str):
                    return value
                else:
                    return str(value)
            except Exception as e:
                logger.error(f"Error extracting {field}: {e}")
                return ""

        subject = extract_value("subject")
        predicate = extract_value("predicate")
        obj = extract_value("object")

        logger.info(f"Processing triplet {idx}: ({subject}, {predicate}, {obj})")

        try:
            # Build messages usando PromptManager
            messages_dict = self.prompt_manager.build_messages(
                'kg_topic_classification',
                subject=subject,
                predicate=predicate,
                object=obj
            )

            # DEBUG: Log dei messaggi
            logger.info(f"Messages dict from PromptManager: {len(messages_dict)} messages")
            if not messages_dict:
                logger.error("PromptManager returned empty messages list!")
                # Proviamo a vedere se il prompt esiste
                available_prompts = self.prompt_manager.list_prompts()
                logger.error(f"Available prompts: {available_prompts}")
                raise ValueError("PromptManager returned empty messages - prompt 'kg_topic_classification' not found")

            # Converti dict in Message objects
            messages = []
            for msg in messages_dict:
                if msg['role'] == 'system':
                    messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))

            # LLM call con structured output
            structured_llm = self.llm.with_structured_output(TopicClassification)
            result = structured_llm.invoke(messages)

            # Log con AgentLogger
            if self.logger:
                self.logger.log_llm_call(
                    messages=[{'role': m.type, 'content': m.content} for m in messages],
                    response=f"broader_topic: {result.broader_topic}\nnarrower_topic: {result.narrower_topic}\nreasoning: {result.reasoning}",
                    model_info={'model': self.llm.model_name}
                )

            logger.info(f"Generated topics for triplet {idx}: {result.broader_topic} → {result.narrower_topic}")

            # Aggiungi metadata alla tripletta (topic LLM generati, non ancora matchati)
            triplet_with_topics = {
                **triplet,
                "broader_topic": result.broader_topic,
                "narrower_topic": result.narrower_topic,
                "classification_reasoning": result.reasoning
            }

            return {
                **state,
                "current_processed_triplet": triplet_with_topics,
                "matched_broader_topic": None,  # Reset per nuova tripletta
                "matched_narrower_topic": None  # Reset per nuova tripletta
            }

        except Exception as e:
            logger.error(f"Error generating topics for triplet {idx}: {str(e)}")
            if self.logger:
                self.logger.log_error(str(e), context=f"Triplet {idx}")

            # In caso di errore, skippiamo questa tripletta incrementando l'indice
            new_errors = state["errors"].copy()
            new_errors.append(f"Triplet {idx}: Failed to generate topics - {str(e)}")
            return {
                **state,
                "errors": new_errors,
                "current_triplet_index": idx + 1
            }

    def _match_broader_topic_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Usa LLM per matchare il broader topic generato con i topic esistenti.

        Returns state con matched_broader_topic impostato.
        """
        triplet = state.get("current_processed_triplet", {})
        if not triplet:
            return state

        generated_broader = triplet.get("broader_topic")
        if not generated_broader:
            return state

        # Recupera tutti i broader topics esistenti
        existing_broader_topics = self.storage.get_all_broader_topics()

        if not existing_broader_topics:
            # Nessun topic esistente, questo sarà il primo
            logger.info(f"No existing broader topics, creating new: {generated_broader}")
            return {
                **state,
                "matched_broader_topic": None  # None = crea nuovo
            }

        # Chiedi all'LLM se c'è un match semantico
        try:
            messages_dict = self.prompt_manager.build_messages(
                'kg_topic_matching',
                new_topic=generated_broader,
                existing_topics=", ".join(existing_broader_topics),
                topic_type="broader"
            )

            messages = []
            for msg in messages_dict:
                if msg['role'] == 'system':
                    messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))

            # Usa invoke normale e parse JSON dalla risposta
            # with_structured_output() non funziona bene con Groq per i boolean
            response = self.llm.invoke(messages)

            import json
            response_text = response.content.strip()

            # Rimuovi markdown code blocks se presenti
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            result_dict = json.loads(response_text)

            # Crea oggetto TopicMatchResult con validazione Pydantic
            result = TopicMatchResult(
                match_found=result_dict['match_found'],
                matched_topic=result_dict.get('matched_topic') or '',  # None → ''
                reasoning=result_dict.get('reasoning') or ''
            )

            if result.match_found:
                logger.info(f"Broader topic match: '{generated_broader}' → '{result.matched_topic}' (reason: {result.reasoning})")
                return {
                    **state,
                    "matched_broader_topic": result.matched_topic
                }
            else:
                logger.info(f"No broader topic match for '{generated_broader}', creating new (reason: {result.reasoning})")
                return {
                    **state,
                    "matched_broader_topic": None
                }

        except Exception as e:
            logger.error(f"Error matching broader topic: {str(e)}")
            # In caso di errore, crea nuovo topic per sicurezza
            return {
                **state,
                "matched_broader_topic": None
            }

    def _match_narrower_topic_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Usa LLM per matchare il narrower topic generato con i topic esistenti
        sotto il broader topic (matchato o nuovo).

        Returns state con matched_narrower_topic impostato.
        """
        triplet = state.get("current_processed_triplet", {})
        if not triplet:
            return state

        generated_narrower = triplet.get("narrower_topic")
        if not generated_narrower:
            return state

        # Determina quale broader topic usare
        matched_broader = state.get("matched_broader_topic")
        generated_broader = triplet.get("broader_topic")

        broader_to_use = matched_broader if matched_broader is not None else generated_broader

        # Recupera i narrower topics per questo broader
        existing_narrower_topics = self.storage.get_narrower_topics_for_broader(broader_to_use)

        if not existing_narrower_topics:
            # Nessun narrower topic esistente sotto questo broader
            logger.info(f"No existing narrower topics under '{broader_to_use}', creating new: {generated_narrower}")
            return {
                **state,
                "matched_narrower_topic": None
            }

        # Chiedi all'LLM se c'è un match semantico
        try:
            messages_dict = self.prompt_manager.build_messages(
                'kg_topic_matching',
                new_topic=generated_narrower,
                existing_topics=", ".join(existing_narrower_topics),
                topic_type="narrower"
            )

            messages = []
            for msg in messages_dict:
                if msg['role'] == 'system':
                    messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))

            # Usa invoke normale e parse JSON dalla risposta
            # with_structured_output() non funziona bene con Groq per i boolean
            response = self.llm.invoke(messages)

            import json
            response_text = response.content.strip()

            # Rimuovi markdown code blocks se presenti
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            result_dict = json.loads(response_text)

            # Crea oggetto TopicMatchResult con validazione Pydantic
            result = TopicMatchResult(
                match_found=result_dict['match_found'],
                matched_topic=result_dict.get('matched_topic') or '',  # None → ''
                reasoning=result_dict.get('reasoning') or ''
            )

            if result.match_found:
                logger.info(f"Narrower topic match: '{generated_narrower}' → '{result.matched_topic}' (reason: {result.reasoning})")
                return {
                    **state,
                    "matched_narrower_topic": result.matched_topic
                }
            else:
                logger.info(f"No narrower topic match for '{generated_narrower}', creating new (reason: {result.reasoning})")
                return {
                    **state,
                    "matched_narrower_topic": None
                }

        except Exception as e:
            logger.error(f"Error matching narrower topic: {str(e)}")
            # In caso di errore, crea nuovo topic per sicurezza
            return {
                **state,
                "matched_narrower_topic": None
            }

    def _store_triplet_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Salva la tripletta nel Knowledge Graph storage usando i topic matchati.
        """
        idx = state["current_triplet_index"]
        triplet = state.get("current_processed_triplet", {})

        if not triplet:
            logger.warning(f"No triplet to store at index {idx}")
            return {
                **state,
                "current_triplet_index": idx + 1
            }

        # Usa i topic matchati (o i generati se non c'è match)
        generated_broader = triplet.get("broader_topic")
        generated_narrower = triplet.get("narrower_topic")

        matched_broader = state.get("matched_broader_topic")
        matched_narrower = state.get("matched_narrower_topic")

        # Determina i topic finali
        final_broader = matched_broader if matched_broader is not None else generated_broader
        final_narrower = matched_narrower if matched_narrower is not None else generated_narrower

        if not final_broader or not final_narrower:
            logger.warning(f"Triplet {idx} missing topics, skipping storage")
            new_errors = state["errors"].copy()
            new_errors.append(f"Triplet {idx}: Missing broader/narrower topics")
            return {
                **state,
                "errors": new_errors,
                "current_triplet_index": idx + 1
            }

        try:
            # Aggiorna la tripletta con i topic finali
            triplet_to_store = {
                **triplet,
                "broader_topic": final_broader,
                "narrower_topic": final_narrower,
                "topic_metadata": {
                    "generated_broader": generated_broader,
                    "generated_narrower": generated_narrower,
                    "matched_broader": matched_broader,
                    "matched_narrower": matched_narrower,
                    "broader_was_merged": matched_broader is not None,
                    "narrower_was_merged": matched_narrower is not None
                }
            }

            # Salva nel storage
            self.storage.add_triplet(triplet_to_store, final_broader, final_narrower)

            logger.info(f"Stored triplet {idx} in KG: {final_broader} → {final_narrower}")

            # Aggiungi alla lista finale
            new_processed = state["processed_triplets"].copy()
            new_processed.append(triplet_to_store)

            return {
                **state,
                "processed_triplets": new_processed,
                "current_triplet_index": idx + 1
            }

        except Exception as e:
            logger.error(f"Error storing triplet {idx}: {str(e)}")
            new_errors = state["errors"].copy()
            new_errors.append(f"Triplet {idx}: Failed to store - {str(e)}")
            return {
                **state,
                "errors": new_errors,
                "current_triplet_index": idx + 1
            }

    # ==================== CONDITIONAL EDGES ====================

    def _check_triplet_valid(self, state: KGBuilderState) -> str:
        """
        Verifica se la tripletta corrente è stata processata correttamente.

        Returns:
            - "valid": Tripletta processata, continua con check_topics
            - "skip": Errore nel processing, salta alla prossima
            - "end": Nessun'altra tripletta da processare
        """
        idx = state["current_triplet_index"]
        total = len(state["triplets"])
        triplet = state.get("current_processed_triplet", {})

        # Se abbiamo finito le triplette
        if idx >= total:
            return "end"

        # Se la tripletta corrente è vuota (errore nel generate_topics)
        if not triplet or "broader_topic" not in triplet:
            # Questa tripletta ha avuto un errore, già gestito in generate_topics
            # L'indice è già stato incrementato, quindi skipiamo al prossimo giro
            return "skip"

        # Tripletta valida, continua normalmente
        return "valid"

    def _should_continue(self, state: KGBuilderState) -> str:
        """Determina se continuare con la prossima tripletta o terminare."""
        idx = state["current_triplet_index"]
        total = len(state["triplets"])

        if idx >= total:
            return "end"
        else:
            return "continue"

    # ==================== PUBLIC API ====================

    def run(self, triplets: List[Dict[str, Any]], ontology_check_enabled: bool = True) -> Dict[str, Any]:
        """
        Processa le triplette e costruisce il Knowledge Graph.

        Args:
            triplets: Lista di triplette da processare
            ontology_check_enabled: Flag per abilitare check ontologico (non ancora usato, preparato per futuro)

        Returns:
            Dict con risultati del processing
        """
        if not triplets:
            return {
                "success": False,
                "error": "No triplets provided",
                "processed_triplets": [],
                "kg_stats": self.storage.get_stats()
            }

        # Inizializza state
        initial_state: KGBuilderState = {
            "triplets": triplets,
            "ontology_check_enabled": ontology_check_enabled,
            "current_triplet_index": 0,
            "current_processed_triplet": {},
            "processed_triplets": [],
            "existing_topics": self.storage.get_all_topics(),
            "errors": [],
            "matched_broader_topic": None,
            "matched_narrower_topic": None
        }

        try:
            # Esegui il grafo
            logger.info(f"Starting KG build for {len(triplets)} triplets")

            final_state = self.graph.invoke(initial_state,{"recursion_limit": 100})

            # Risultati
            result = {
                "success": True,
                "processed_triplets": final_state["processed_triplets"],
                "kg_stats": self.storage.get_stats(),
                "errors": final_state["errors"]
            }

            logger.info(f"KG build completed: {result['kg_stats']}")

            # Log summary
            if self.logger:
                self.logger.log_summary()

            return result

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error during KG build: {str(e)}\n{error_details}")

            if self.logger:
                self.logger.log_error(f"{str(e)}\n\nStack trace:\n{error_details}", context="KG Build")

            return {
                "success": False,
                "error": f"{str(e)}\n\nStack trace:\n{error_details}",
                "processed_triplets": [],
                "kg_stats": self.storage.get_stats()
            }

    def get_storage(self) -> InMemoryKnowledgeGraph:
        """Ritorna lo storage corrente (utile per debugging e visualizzazione)."""
        return self.storage
