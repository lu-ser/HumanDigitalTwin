"""
LangGraph pipeline per costruire il Knowledge Graph da triplette validate.

Workflow:
1. Prende triplette (opzionalmente validate con ontology)
2. Genera broad_topic e narrow_topic per ogni tripletta (LLM)
3. Verifica se i topic esistono già nel KG
4. Salva tripletta + metadata nel Knowledge Graph
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import operator
import logging

logger = logging.getLogger(__name__)


# ==================== PYDANTIC MODELS ====================

class TopicClassification(BaseModel):
    """Classificazione di una tripletta in broad e narrow topics."""
    broader_topic: str = Field(description="Categoria ampia (es. 'Health', 'Finance', 'Social')")
    narrower_topic: str = Field(description="Sottocategoria specifica (es. 'Heart Rate', 'Expenses', 'Friends')")
    reasoning: str = Field(description="Breve spiegazione della classificazione")


# ==================== STATE DEFINITION ====================

class KGBuilderState(TypedDict):
    """
    State per il Knowledge Graph Builder.

    Campi:
    - triplets: Lista di triplette da processare
    - ontology_check_enabled: Flag per abilitare/disabilitare check ontologico
    - current_triplet_index: Indice della tripletta corrente
    - processed_triplets: Lista di triplette processate con metadata
    - existing_topics: Set di coppie (broader, narrower) già esistenti nel KG
    - errors: Lista di errori durante il processing
    """
    triplets: List[Dict[str, Any]]
    ontology_check_enabled: bool
    current_triplet_index: int
    processed_triplets: Annotated[List[Dict[str, Any]], operator.add]
    existing_topics: Dict[str, List[str]]  # {broader_topic: [narrower_topic1, narrower_topic2, ...]}
    errors: Annotated[List[str], operator.add]


# ==================== IN-MEMORY STORAGE (TODO: Replace with Neo4j) ====================

class InMemoryKnowledgeGraph:
    """
    Storage temporaneo in-memory per il Knowledge Graph.

    IMPORTANTE: Questa classe è un placeholder temporaneo.
    Sarà sostituita con Neo4j nelle prossime iterazioni.
    """

    def __init__(self):
        # Structure: {broader_topic: {narrower_topic: [triplets]}}
        self.graph: Dict[str, Dict[str, List[Dict]]] = {}

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


# ==================== KNOWLEDGE GRAPH BUILDER ====================

class KnowledgeGraphBuilder:
    """
    LangGraph pipeline per costruire il Knowledge Graph.
    """

    def __init__(self, llm, storage: InMemoryKnowledgeGraph = None):
        """
        Inizializza il builder.

        Args:
            llm: Istanza del modello LLM
            storage: Storage per il KG (default: InMemoryKnowledgeGraph)
        """
        self.llm = llm
        self.storage = storage or InMemoryKnowledgeGraph()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Costruisce il grafo LangGraph."""
        workflow = StateGraph(KGBuilderState)

        # Nodi
        workflow.add_node("generate_topics", self._generate_topics_node)
        workflow.add_node("check_topics_exist", self._check_topics_exist_node)
        workflow.add_node("store_triplet", self._store_triplet_node)

        # Entry point
        workflow.set_entry_point("generate_topics")

        # Edges
        workflow.add_edge("generate_topics", "check_topics_exist")
        workflow.add_edge("check_topics_exist", "store_triplet")

        # Conditional: continua con prossima tripletta o termina
        workflow.add_conditional_edges(
            "store_triplet",
            self._should_continue,
            {
                "continue": "generate_topics",
                "end": END
            }
        )

        return workflow.compile()

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
            value = triplet.get(field, "")
            if isinstance(value, dict):
                return value.get("value", "")
            elif isinstance(value, str):
                return value
            else:
                return str(value)

        subject = extract_value("subject")
        predicate = extract_value("predicate")
        obj = extract_value("object")

        # Prompt per classificazione
        system_prompt = """You are a knowledge graph topic classifier.

Given a triplet (subject, predicate, object), classify it into:
1. **broader_topic**: A broad category (e.g., "Health", "Finance", "Social", "Work", "Hobbies")
2. **narrower_topic**: A specific subcategory (e.g., "Heart Rate", "Expenses", "Friends", "Projects")

Rules:
- Use clear, concise category names (1-3 words)
- broader_topic should be general and reusable across multiple triplets
- narrower_topic should be specific to this type of information
- Provide a brief reasoning for your classification

Examples:
- Triplet: ("John", "has_friend", "Mary") → broader="Social", narrower="Friendships"
- Triplet: ("User", "heart_rate", "72 bpm") → broader="Health", narrower="Vital Signs"
- Triplet: ("Account", "balance", "$1500") → broader="Finance", narrower="Bank Accounts"
"""

        user_prompt = f"""Classify this triplet:

Subject: {subject}
Predicate: {predicate}
Object: {obj}

Provide the classification in the specified format."""

        try:
            # LLM call con structured output
            structured_llm = self.llm.with_structured_output(TopicClassification)

            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            logger.info(f"Generated topics for triplet {idx}: {result.broader_topic} → {result.narrower_topic}")

            # Aggiungi metadata alla tripletta
            triplet_with_topics = {
                **triplet,
                "broader_topic": result.broader_topic,
                "narrower_topic": result.narrower_topic,
                "classification_reasoning": result.reasoning
            }

            return {
                **state,
                "processed_triplets": [triplet_with_topics]
            }

        except Exception as e:
            logger.error(f"Error generating topics for triplet {idx}: {str(e)}")
            return {
                **state,
                "errors": [f"Triplet {idx}: Failed to generate topics - {str(e)}"]
            }

    def _check_topics_exist_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Verifica se i topic esistono già nel KG.

        Implementa la logica da Instructions.md:
        - Se broader_topic esiste: usa quello, altrimenti crea
        - Se narrower_topic esiste sotto quel broader: usa quello, altrimenti crea
        """
        idx = state["current_triplet_index"]
        processed = state["processed_triplets"]

        if not processed or idx >= len(processed):
            return state

        triplet = processed[-1]  # Ultima tripletta processata
        broader = triplet.get("broader_topic")
        narrower = triplet.get("narrower_topic")

        if not broader or not narrower:
            return state

        # Check esistenza
        broader_exists = self.storage.topic_exists(broader)
        narrower_exists = self.storage.topic_exists(broader, narrower)

        # Aggiungi metadata
        triplet["topic_metadata"] = {
            "broader_exists": broader_exists,
            "narrower_exists": narrower_exists,
            "action": "reuse" if narrower_exists else "create_narrower" if broader_exists else "create_both"
        }

        logger.info(f"Topic check: broader={broader} (exists={broader_exists}), narrower={narrower} (exists={narrower_exists})")

        return state

    def _store_triplet_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Salva la tripletta nel Knowledge Graph storage.
        """
        idx = state["current_triplet_index"]
        processed = state["processed_triplets"]

        if not processed or idx >= len(processed):
            return state

        triplet = processed[-1]
        broader = triplet.get("broader_topic")
        narrower = triplet.get("narrower_topic")

        if not broader or not narrower:
            logger.warning(f"Triplet {idx} missing topics, skipping storage")
            return {
                **state,
                "errors": [f"Triplet {idx}: Missing broader/narrower topics"],
                "current_triplet_index": idx + 1
            }

        try:
            # Salva nel storage
            self.storage.add_triplet(triplet, broader, narrower)

            logger.info(f"Stored triplet {idx} in KG")

            return {
                **state,
                "current_triplet_index": idx + 1
            }

        except Exception as e:
            logger.error(f"Error storing triplet {idx}: {str(e)}")
            return {
                **state,
                "errors": [f"Triplet {idx}: Failed to store - {str(e)}"],
                "current_triplet_index": idx + 1
            }

    # ==================== CONDITIONAL EDGES ====================

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
            "processed_triplets": [],
            "existing_topics": self.storage.get_all_topics(),
            "errors": []
        }

        try:
            # Esegui il grafo
            logger.info(f"Starting KG build for {len(triplets)} triplets")
            final_state = self.graph.invoke(initial_state)

            # Risultati
            result = {
                "success": True,
                "processed_triplets": final_state["processed_triplets"],
                "kg_stats": self.storage.get_stats(),
                "errors": final_state["errors"]
            }

            logger.info(f"KG build completed: {result['kg_stats']}")

            return result

        except Exception as e:
            logger.error(f"Error during KG build: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_triplets": [],
                "kg_stats": self.storage.get_stats()
            }

    def get_storage(self) -> InMemoryKnowledgeGraph:
        """Ritorna lo storage corrente (utile per debugging e visualizzazione)."""
        return self.storage
