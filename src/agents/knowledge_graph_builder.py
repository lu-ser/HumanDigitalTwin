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
from pydantic import BaseModel, Field
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
    - current_processed_triplet: Tripletta corrente in processing (sovrascritta ad ogni iterazione)
    - processed_triplets: Lista finale di triplette processate (accumulata)
    - existing_topics: Set di coppie (broader, narrower) già esistenti nel KG
    - errors: Lista di errori durante il processing
    """
    triplets: List[Dict[str, Any]]
    ontology_check_enabled: bool
    current_triplet_index: int
    current_processed_triplet: Dict[str, Any]  # Triplet corrente (non accumula)
    processed_triplets: List[Dict[str, Any]]  # Lista finale (gestita manualmente)
    existing_topics: Dict[str, List[str]]  # {broader_topic: [narrower_topic1, narrower_topic2, ...]}
    errors: List[str]  # Lista errori (gestita manualmente)


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
        """Costruisce il grafo LangGraph."""
        workflow = StateGraph(KGBuilderState)

        # Nodi
        workflow.add_node("generate_topics", self._generate_topics_node)
        workflow.add_node("check_topics_exist", self._check_topics_exist_node)
        workflow.add_node("store_triplet", self._store_triplet_node)

        # Entry point
        workflow.set_entry_point("generate_topics")

        # Conditional da generate_topics: se triplet vuoto skippa, altrimenti continua
        workflow.add_conditional_edges(
            "generate_topics",
            self._check_triplet_valid,
            {
                "valid": "check_topics_exist",
                "skip": "generate_topics",  # Retry con prossima tripletta
                "end": END  # Fine se non ci sono più triplette
            }
        )

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

            # Aggiungi metadata alla tripletta
            triplet_with_topics = {
                **triplet,
                "broader_topic": result.broader_topic,
                "narrower_topic": result.narrower_topic,
                "classification_reasoning": result.reasoning
            }

            return {
                **state,
                "current_processed_triplet": triplet_with_topics
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

    def _check_topics_exist_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Verifica se i topic esistono già nel KG.

        Implementa la logica da Instructions.md:
        - Se broader_topic esiste: usa quello, altrimenti crea
        - Se narrower_topic esiste sotto quel broader: usa quello, altrimenti crea
        """
        triplet = state.get("current_processed_triplet", {})

        if not triplet:
            return state

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

        return {
            **state,
            "current_processed_triplet": triplet
        }

    def _store_triplet_node(self, state: KGBuilderState) -> KGBuilderState:
        """
        Salva la tripletta nel Knowledge Graph storage e la aggiunge alla lista finale.
        """
        idx = state["current_triplet_index"]
        triplet = state.get("current_processed_triplet", {})

        if not triplet:
            logger.warning(f"No triplet to store at index {idx}")
            return {
                **state,
                "current_triplet_index": idx + 1
            }

        broader = triplet.get("broader_topic")
        narrower = triplet.get("narrower_topic")

        if not broader or not narrower:
            logger.warning(f"Triplet {idx} missing topics, skipping storage")
            new_errors = state["errors"].copy()
            new_errors.append(f"Triplet {idx}: Missing broader/narrower topics")
            return {
                **state,
                "errors": new_errors,
                "current_triplet_index": idx + 1
            }

        try:
            # Salva nel storage
            self.storage.add_triplet(triplet, broader, narrower)

            logger.info(f"Stored triplet {idx} in KG")

            # Aggiungi alla lista finale
            new_processed = state["processed_triplets"].copy()
            new_processed.append(triplet)

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
