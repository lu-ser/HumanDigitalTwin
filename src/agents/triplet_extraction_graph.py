"""
LangGraph pipeline per estrazione multi-stage di triplette da testo con augmentation IoT.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import operator


# Pydantic models per structured output
class TripletEntity(BaseModel):
    """Entità con valore e tipo."""
    value: str = Field(description="Valore dell'entità (es. 'Marco')")
    type: str = Field(description="Tipo/classe dell'entità (es. 'Person', 'Location', 'Integer')")


class Triplet(BaseModel):
    """Singola tripletta RDF con tipizzazione (matrice 2x3: instance + type)."""
    subject: TripletEntity = Field(description="Soggetto della tripletta con tipo")
    predicate: TripletEntity = Field(description="Predicato/relazione della tripletta con tipo")
    object: TripletEntity = Field(description="Oggetto della tripletta con tipo")


class TripletList(BaseModel):
    """Lista di triplette."""
    triplets: List[Triplet] = Field(description="Lista di triplette RDF estratte")

    def filter_valid(self) -> List[Triplet]:
        """Filtra solo le triplette valide (con subject, predicate e object non vuoti)."""
        return [
            t for t in self.triplets
            if t.subject.value and t.predicate.value and t.object.value
        ]


class GraphState(TypedDict):
    """
    State del grafo per l'estrazione di triplette.

    Campi:
    - input_text: Testo originale da processare
    - chunk_size: Dimensione dei chunk (numero di caratteri)
    - chunks: Lista di chunk di testo
    - current_chunk_index: Indice del chunk corrente in elaborazione
    - previous_summary: Summary del chunk precedente
    - triplets: Lista accumulata di tutte le triplette estratte
    - augmented_triplets: Triplette aggiunte dall'augmentation (text + IoT)
    - final_triplets: Triplette finali (estratte + augmented)
    - error: Eventuale errore durante l'elaborazione

    IoT ReAct Loop:
    - iot_should_explore: Flag per decidere se iniziare esplorazione IoT
    - iot_react_iteration: Contatore iterazioni ReAct (max 5)
    - iot_data_collected: Dati IoT raccolti durante l'esplorazione
    - iot_reasoning_history: Storia del reasoning dell'agent IoT
    - iot_conversation: Conversazione completa con LLM (per context in loop)
    """
    input_text: str
    chunk_size: int
    chunks: List[str]
    current_chunk_index: int
    previous_summary: Optional[str]
    triplets: Annotated[List[Dict[str, str]], operator.add]
    augmented_triplets: Annotated[List[Dict[str, str]], operator.add]
    final_triplets: List[Dict[str, str]]
    error: Optional[str]

    # IoT ReAct Loop State
    iot_should_explore: bool
    iot_react_iteration: int
    iot_data_collected: Dict[str, Any]
    iot_reasoning_history: Annotated[List[str], operator.add]
    iot_conversation: List[Any]  # Lista di messaggi LangChain

    # Validation Guardrail State
    validation_should_run: bool
    validation_iteration: int
    validation_reasoning: Annotated[List[str], operator.add]
    validated_triplets: List[Dict[str, str]]  # Triplette dopo validazione
    removed_triplets: Annotated[List[Dict[str, str]], operator.add]  # Triplette rimosse con reason


class TripletExtractionGraph:
    """
    Grafo LangGraph per estrazione multi-stage di triplette.

    Pipeline:
    1. Chunking del testo
    2. Per ogni chunk:
       - Crea summary del chunk precedente (se esiste)
       - Estrai triplette dal chunk corrente
    3. Analisi per augmentation IoT
    4. Augmentation delle triplette con dati IoT (se necessario)
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        mcp_base_url: str = "http://localhost:8000",
        temperature: float = 0.3,
        enable_logging: bool = False
    ):
        """
        Inizializza il grafo di estrazione triplette.

        Args:
            llm_api_key: API key per Groq
            llm_model: Modello LLM da utilizzare
            mcp_base_url: URL del server MCP
            temperature: Temperature per l'LLM
            enable_logging: Abilita logging dettagliato
        """
        self.llm = ChatGroq(
            groq_api_key=llm_api_key,
            model_name=llm_model,
            temperature=temperature
        )
        self.mcp_base_url = mcp_base_url
        self.enable_logging = enable_logging

        # Logger
        if self.enable_logging:
            from src.utils import get_logger
            self.logger = get_logger()
        else:
            self.logger = None

        # Prompt Manager
        from src.prompts import PromptManager
        self.prompt_manager = PromptManager()

        # MCP Tools per l'agent
        from src.mcp.mcp_tools import get_mcp_tools
        self.mcp_tools = get_mcp_tools(self.mcp_base_url)

        # Costruisci il grafo
        self.graph = self._build_graph()

        # Salva il diagramma Mermaid del grafo
        self._save_graph_diagram()

    def _build_graph(self) -> StateGraph:
        """Costruisce il grafo LangGraph con ReAct loop per IoT augmentation."""
        workflow = StateGraph(GraphState)

        # === NODI DEL GRAFO ===
        # Chunking & Extraction
        workflow.add_node("chunk_text", self._chunk_text_node)
        workflow.add_node("extract_triplets", self._extract_triplets_node)
        workflow.add_node("summarize_chunk", self._summarize_chunk_node)

        # Text Augmentation
        workflow.add_node("text_augmentation", self._text_augmentation_node)

        # IoT ReAct Loop (3 nodi)
        workflow.add_node("iot_decide", self._iot_decide_node)              # Decisione iniziale
        workflow.add_node("iot_react", self._iot_react_node)                # Iterazione ReAct (loop)
        workflow.add_node("iot_generate", self._iot_generate_triplets_node) # Genera triplette finali

        # Validation Guardrail (2 nodi)
        workflow.add_node("validation_decide", self._validation_decide_node)    # Decisione validazione
        workflow.add_node("validation_iterate", self._validation_iterate_node)  # Iterazione validazione (loop)

        # Finalizzazione
        workflow.add_node("finalize", self._finalize_node)

        # === FLUSSO DEL GRAFO ===
        workflow.set_entry_point("chunk_text")

        # Chunk → Extract
        workflow.add_edge("chunk_text", "extract_triplets")

        # Extract → Summarize OR Text Augmentation (loop chunks)
        workflow.add_conditional_edges(
            "extract_triplets",
            self._should_continue_chunks,
            {
                "summarize": "summarize_chunk",
                "augment": "text_augmentation"
            }
        )

        # Summarize → Extract (loop back)
        workflow.add_edge("summarize_chunk", "extract_triplets")

        # Text Augmentation → IoT Decision
        workflow.add_edge("text_augmentation", "iot_decide")

        # IoT Decision → Explore OR Skip
        workflow.add_conditional_edges(
            "iot_decide",
            self._should_use_iot,
            {
                "explore": "iot_react",           # Inizia ReAct loop
                "skip": "validation_decide"       # Salta IoT, vai a validazione
            }
        )

        # IoT ReAct → Continue (self-loop) OR Finish
        workflow.add_conditional_edges(
            "iot_react",
            self._should_continue_react,
            {
                "continue": "iot_react",      # Loop: chiama altri tools
                "finish": "iot_generate"      # Genera triplette IoT
            }
        )

        # IoT Generate → Validation Decision
        workflow.add_edge("iot_generate", "validation_decide")

        # Validation Decision → Validate OR Skip
        workflow.add_conditional_edges(
            "validation_decide",
            self._should_validate,
            {
                "validate": "validation_iterate",  # Inizia validation loop
                "skip": "finalize"                  # Salta validazione
            }
        )

        # Validation Iterate → Continue (self-loop) OR Finish
        workflow.add_conditional_edges(
            "validation_iterate",
            self._should_continue_validation,
            {
                "continue": "validation_iterate",  # Loop: refina ancora
                "finish": "finalize"                # Validazione completa
            }
        )

        # Finalize → END
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _save_graph_diagram(self) -> None:
        """Salva il diagramma Mermaid del grafo come PNG."""
        try:
            from pathlib import Path

            # Crea la cartella assets se non esiste
            assets_dir = Path(__file__).parent.parent.parent / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Genera e salva il diagramma
            output_path = assets_dir / "triplet_extraction_graph.png"
            png_data = self.graph.get_graph().draw_mermaid_png()

            with open(output_path, "wb") as f:
                f.write(png_data)

            msg = f"Graph diagram saved to: {output_path}"
            if self.logger:
                self.logger.console.print(f"[dim]{msg}[/dim]")
            else:
                print(msg)

        except Exception as e:
            msg = f"Warning: Could not save graph diagram: {str(e)}"
            if self.logger:
                self.logger.console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(msg)

    def _chunk_text_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 1: Divide il testo in chunk.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con i chunk creati
        """
        text = state["input_text"]
        chunk_size = state.get("chunk_size", 1000)

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold blue]═══ Chunking Node ═══[/bold blue]")
            self.logger.console.print(f"[dim]Text length: {len(text)} chars, Chunk size: {chunk_size}[/dim]")

        # Chunking semplice per caratteri
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)

        if self.logger:
            self.logger.console.print(f"[green]Created {len(chunks)} chunks[/green]")

        return {
            "chunks": chunks,
            "current_chunk_index": 0,
            "previous_summary": None,
            "triplets": []
        }

    def _extract_triplets_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 2: Estrae triplette dal chunk corrente.

        Usa il contesto della summary del chunk precedente se disponibile.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con le triplette estratte
        """
        chunks = state["chunks"]
        current_index = state["current_chunk_index"]
        previous_summary = state.get("previous_summary")

        if current_index >= len(chunks):
            return {}

        current_chunk = chunks[current_index]

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold cyan]═══ Triplet Extraction Node - Chunk {current_index + 1}/{len(chunks)} ═══[/bold cyan]")

        # Costruisci il prompt usando PromptManager
        context_text = ""
        if previous_summary:
            context_text = f"Context from previous chunk:\n{previous_summary}\n"
            if self.logger:
                self.logger.console.print(f"[dim]Using context from previous chunk[/dim]")

        messages = self.prompt_manager.build_messages(
            'triplet_extraction_chunk',
            context=context_text,
            chunk=current_chunk
        )

        # Converti in formato Langchain
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            # Usa structured output con Pydantic (json_mode for better compatibility)
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(lc_messages)

            # Filtra solo triplette valide e converti da Pydantic a dict (matrice 2x3)
            valid_triplets = response.filter_valid()
            new_triplets = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid_triplets
            ]

            # Log risposta
            if self.logger:
                import json
                filtered_count = len(response.triplets) - len(valid_triplets)
                if filtered_count > 0:
                    self.logger.console.print(f"[dim]Filtered out {filtered_count} invalid triplets with null/empty values[/dim]")
                self.logger.log_agent_response(f"Extracted {len(new_triplets)} triplets:\n{json.dumps(new_triplets, indent=2)}")

            return {
                "triplets": new_triplets,
                "current_chunk_index": current_index + 1
            }

        except Exception as e:
            error_msg = f"Errore estrazione triplette chunk {current_index}: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "Triplet Extraction Node")
            else:
                print(error_msg)
            return {
                "error": f"Errore estrazione triplette: {str(e)}",
                "current_chunk_index": current_index + 1,
                "triplets": []
            }

    def _summarize_chunk_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 3: Crea una summary del chunk appena processato.

        Questa summary sarà usata come contesto per il prossimo chunk.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con la summary
        """
        chunks = state["chunks"]
        current_index = state["current_chunk_index"]

        # Il chunk precedente è quello appena processato
        previous_chunk_index = current_index - 1

        if previous_chunk_index < 0 or previous_chunk_index >= len(chunks):
            return {"previous_summary": None}

        previous_chunk = chunks[previous_chunk_index]

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold magenta]═══ Summarization Node - Chunk {previous_chunk_index + 1} ═══[/bold magenta]")

        # Usa PromptManager
        messages = self.prompt_manager.build_messages(
            'triplet_summarization',
            chunk=previous_chunk
        )

        # Converti in formato Langchain
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            response = self.llm.invoke(lc_messages)
            summary = response.content.strip()

            # Log risposta
            if self.logger:
                self.logger.log_agent_response(f"Summary:\n{summary}")

            return {"previous_summary": summary}

        except Exception as e:
            error_msg = f"Errore summarization: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "Summarization Node")
            return {
                "error": error_msg,
                "previous_summary": None
            }

    def _should_continue_chunks(self, state: GraphState) -> str:
        """
        Conditional edge: decide se continuare con altri chunk o passare ad augmentation.

        Args:
            state: Stato corrente del grafo

        Returns:
            "summarize" se ci sono altri chunk da processare, "augment" altrimenti
        """
        current_index = state["current_chunk_index"]
        chunks = state["chunks"]

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_chunks[/bold cyan]")
            self.logger.console.print(f"[dim]Chunk {current_index}/{len(chunks)}[/dim]")

        if current_index < len(chunks):
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: summarize (more chunks to process)[/green]")
            return "summarize"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: augment (all chunks processed, starting augmentation)[/yellow]")
            return "augment"

    def _text_augmentation_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4: Augmenta le triplette estratte con relazioni implicite e correzioni.

        Arricchisce il knowledge graph correggendo inconsistenze,
        aggiungendo relazioni implicite e rendendo i predicati coerenti.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con le triplette augmented (via text reasoning)
        """
        triplets = state.get("triplets", [])

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold yellow]═══ Text Augmentation Node ═══[/bold yellow]")
            self.logger.console.print(f"[dim]Refining and enriching {len(triplets)} extracted triplets[/dim]")

        if not triplets:
            if self.logger:
                self.logger.console.print(f"[dim]No triplets to augment, skipping[/dim]")
            return {"augmented_triplets": []}

        triplets_str = "\n".join([
            f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for t in triplets
        ])

        # Usa PromptManager
        messages = self.prompt_manager.build_messages(
            'text_augmentation',
            triplets=triplets_str
        )

        # Converti in formato Langchain
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            # Usa structured output con Pydantic (json_mode)
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(lc_messages)

            # Filtra solo triplette valide e converti da Pydantic a dict (matrice 2x3)
            valid_triplets = response.filter_valid()
            augmented = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid_triplets
            ]

            # Log risposta
            if self.logger:
                import json
                filtered_count = len(response.triplets) - len(valid_triplets)
                if filtered_count > 0:
                    self.logger.console.print(f"[dim]Filtered out {filtered_count} invalid triplets with null/empty values[/dim]")
                self.logger.log_agent_response(f"Generated {len(augmented)} augmented triplets:\n{json.dumps(augmented, indent=2)}")

            return {"augmented_triplets": augmented}

        except Exception as e:
            error_msg = f"Errore text augmentation: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "Text Augmentation Node")
            return {
                "error": error_msg,
                "augmented_triplets": []
            }

    def _iot_decide_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4a: Decide se utilizzare dati IoT per augmentation.

        Analizza le triplette text-augmented e decide se ha senso
        arricchirle con dati da sensori IoT.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con la decisione (iot_should_explore)
        """
        augmented_triplets = state.get("augmented_triplets", [])

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold magenta]═══ IoT Decision Node ═══[/bold magenta]")
            self.logger.console.print(f"[dim]Analyzing {len(augmented_triplets)} triplets to decide if IoT data could add value[/dim]")

        if not augmented_triplets:
            if self.logger:
                self.logger.console.print(f"[yellow]No triplets to augment, skipping IoT exploration[/yellow]")
            return {
                "iot_should_explore": False,
                "iot_react_iteration": 0,
                "iot_data_collected": {},
                "iot_conversation": []
            }

        # Prepara le triplette per il prompt
        triplets_str = "\n".join([
            f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for t in augmented_triplets
        ])

        # Usa PromptManager per il prompt di decisione
        messages = self.prompt_manager.build_messages(
            'iot_decide',
            triplets=triplets_str
        )

        # Converti in formato Langchain
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            # Chiedi all'LLM se vale la pena esplorare IoT
            response = self.llm.invoke(lc_messages)
            decision_text = response.content.strip().upper()

            # Parsing semplice: cerca YES/SI o NO
            should_explore = any(keyword in decision_text for keyword in ["YES", "SI", "SÌ", "USEFUL", "RELEVANT"])

            # Log risposta
            if self.logger:
                decision_emoji = "✅" if should_explore else "❌"
                self.logger.log_agent_response(f"{decision_emoji} Decision: {'EXPLORE IoT data' if should_explore else 'SKIP IoT exploration'}\n\nReasoning:\n{response.content}")

            return {
                "iot_should_explore": should_explore,
                "iot_react_iteration": 0,
                "iot_data_collected": {},
                "iot_reasoning_history": [f"Initial Decision: {'Explore' if should_explore else 'Skip'}"],
                "iot_conversation": []  # Inizia vuoto, il ReAct node costruirà la sua conversazione
            }

        except Exception as e:
            error_msg = f"Errore IoT decision: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "IoT Decision Node")
            return {
                "error": error_msg,
                "iot_should_explore": False,
                "iot_react_iteration": 0,
                "iot_data_collected": {},
                "iot_conversation": []
            }

    def _iot_react_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4b: Esegue UNA iterazione del ReAct loop per IoT augmentation.

        Segue il pattern Think → Act → Observe:
        1. THINK: L'agent analizza i dati già raccolti e decide la prossima azione
        2. ACT: Chiama UN tool MCP (list_devices, get_latest_value, etc.)
        3. OBSERVE: Riceve il risultato e aggiorna lo state

        Il loop continua finché l'agent decide di fermarsi o max_iterations.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con:
            - iot_conversation: conversazione aggiornata
            - iot_data_collected: dati aggiornati
            - iot_react_iteration: contatore incrementato
            - iot_reasoning_history: reasoning dell'iterazione
        """
        # Recupera lo state del ReAct loop
        iteration = state.get("iot_react_iteration", 0)
        conversation = state.get("iot_conversation", [])
        data_collected = state.get("iot_data_collected", {})
        augmented_triplets = state.get("augmented_triplets", [])

        # Log inizio iterazione
        if self.logger:
            self.logger.console.print(f"\n[bold green]🔄 IoT ReAct Iteration {iteration + 1}/5[/bold green]")

            # Mostra i dati già raccolti
            if data_collected:
                from rich.panel import Panel
                import json
                data_panel = Panel(
                    json.dumps(data_collected, indent=2),
                    title="[yellow]📊 Data Collected So Far[/yellow]",
                    border_style="yellow"
                )
                self.logger.console.print(data_panel)

        # Se è la prima iterazione, inizializza la conversazione
        if not conversation:
            triplets_str = "\n".join([
                f"- {t.get('subject', '')} {t.get('predicate', '')} {t.get('object', '')}"
                for t in augmented_triplets
            ])

            # Usa il prompt iot_react_iteration
            messages = self.prompt_manager.build_messages(
                'iot_react_iteration',
                triplets=triplets_str,
                data_collected=json.dumps(data_collected, indent=2) if data_collected else "None"
            )

            # Converti in formato Langchain
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content')
                if role == 'system':
                    conversation.append(SystemMessage(content=content))
                else:
                    conversation.append(HumanMessage(content=content))

        # Log prompt (solo prima iterazione)
        if self.logger and iteration == 0:
            self.logger.log_llm_call(
                [{"role": m.type, "content": m.content} for m in conversation],
                "",
                {"model": self.llm.model_name, "provider": "Groq", "with_tools": len(self.mcp_tools)}
            )

        try:
            from langchain_core.messages import ToolMessage
            import json

            # Chiama LLM con i tools disponibili
            llm_with_tools = self.llm.bind_tools(self.mcp_tools)
            response = llm_with_tools.invoke(conversation)

            # 🤔 THINK: Log reasoning dell'agent
            reasoning = ""
            if hasattr(response, 'content') and response.content:
                reasoning = response.content
                if self.logger:
                    from rich.panel import Panel
                    think_panel = Panel(
                        reasoning,
                        title="[cyan]🤔 THINK: Agent Reasoning[/cyan]",
                        border_style="cyan",
                        padding=(1, 2)
                    )
                    self.logger.console.print(think_panel)

            # Aggiungi la risposta AI alla conversazione
            conversation.append(response)

            # Se ci sono tool calls: ACT + OBSERVE
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 🔧 ACT: Esegui i tool calls
                if self.logger:
                    self.logger.console.print(f"\n[bold yellow]🔧 ACT: Agent calling {len(response.tool_calls)} tool(s)[/bold yellow]")

                for idx, tool_call in enumerate(response.tool_calls, 1):
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    if self.logger:
                        from rich.panel import Panel
                        from rich.syntax import Syntax

                        # Mostra la chiamata
                        self.logger.console.print(f"\n[bold cyan]📞 Tool #{idx}: {tool_name}[/bold cyan]")
                        args_json = json.dumps(tool_args, indent=2)
                        args_syntax = Syntax(args_json, "json", theme="monokai", line_numbers=False)
                        args_panel = Panel(
                            args_syntax,
                            title=f"[yellow]Parameters[/yellow]",
                            border_style="yellow",
                            padding=(0, 1)
                        )
                        self.logger.console.print(args_panel)

                    # Trova ed esegui il tool
                    selected_tool = None
                    for tool in self.mcp_tools:
                        if tool.name == tool_name:
                            selected_tool = tool
                            break

                    if selected_tool:
                        tool_result = selected_tool.invoke(tool_args)

                        # 👁️ OBSERVE: Mostra il risultato
                        if self.logger:
                            result_str = str(tool_result)
                            try:
                                result_json = json.loads(result_str) if result_str.startswith(('{', '[')) else result_str
                                if isinstance(result_json, (dict, list)):
                                    result_formatted = json.dumps(result_json, indent=2)
                                    result_syntax = Syntax(result_formatted, "json", theme="monokai", line_numbers=False)
                                else:
                                    result_syntax = result_str
                            except:
                                result_syntax = result_str

                            observe_panel = Panel(
                                result_syntax,
                                title=f"[green]👁️  OBSERVE: Result[/green]",
                                border_style="green",
                                padding=(0, 1)
                            )
                            self.logger.console.print(observe_panel)

                            # Log nel file
                            self.logger.log_tool_call(tool_name, tool_args, str(tool_result))

                        # Aggiungi risultato alla conversazione
                        conversation.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call['id']
                            )
                        )

                        # Salva dati raccolti nello state
                        data_collected[f"{tool_name}_{iteration}"] = {
                            "tool": tool_name,
                            "args": tool_args,
                            "result": str(tool_result)
                        }

                # Incrementa iterazione e continua il loop
                return {
                    "iot_conversation": conversation,
                    "iot_data_collected": data_collected,
                    "iot_react_iteration": iteration + 1,
                    "iot_reasoning_history": [reasoning] if reasoning else []
                }

            else:
                # ✅ FINISH: Nessun tool chiamato, l'agent ha deciso di fermarsi
                if self.logger:
                    self.logger.console.print(f"\n[bold green]✅ DECIDE: Agent decided to FINISH (no more tools needed)[/bold green]")

                # L'agent ha finito, non continuare il loop
                return {
                    "iot_conversation": conversation,
                    "iot_react_iteration": iteration + 1,
                    "iot_reasoning_history": [reasoning] if reasoning else [],
                    "iot_should_explore": False  # Stop il loop
                }

        except Exception as e:
            error_msg = f"Errore IoT ReAct iteration: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "IoT ReAct Node")
            return {
                "error": error_msg,
                "iot_should_explore": False  # Stop il loop in caso di errore
            }

    def _iot_generate_triplets_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4c: Genera triplette IoT finali dai dati raccolti.

        Dopo aver completato il ReAct loop, questo nodo analizza tutti i dati
        raccolti e genera le triplette finali da aggiungere al knowledge graph.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con augmented_triplets (IoT)
        """
        conversation = state.get("iot_conversation", [])
        data_collected = state.get("iot_data_collected", {})
        augmented_triplets = state.get("augmented_triplets", [])

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold blue]═══ IoT Triplet Generation Node ═══[/bold blue]")
            self.logger.console.print(f"[dim]Generating triplets from collected IoT data[/dim]")

        if not data_collected:
            if self.logger:
                self.logger.console.print(f"[yellow]No IoT data collected, skipping triplet generation[/yellow]")
            # Non ritornare augmented_triplets: [] perché cancellerebbe le text-augmented!
            # Ritorna {} per non modificare lo state
            return {}

        try:
            import json

            # Prepara il prompt per la generazione finale
            triplets_str = "\n".join([
                f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
                for t in augmented_triplets
            ])

            data_summary = json.dumps(data_collected, indent=2)

            # Usa il prompt per generare le triplette finali
            final_prompt = HumanMessage(content=f"""
Based on the IoT data collected, generate NEW triplets that connect the entities to sensor data.

Original Triplets:
{triplets_str}

IoT Data Collected:
{data_summary}

Generate triplets using predicates like:
- hasHeartRate, hasTemperature, hasLocation
- measuredAt, recordedAt, detectedActivity
- associatedWith, relatedTo

**IMPORTANT**: Each entity MUST have both "value" and "type" fields.
- People/devices: type = "Person" or "Thing"
- Sensor values: type = "Number" or "Thing"
- Timestamps: type = "DateTime"
- Locations: type = "Place"
- All predicates: type = "Relationship"

Return ONLY valid triplets in JSON format:
{{"triplets": [{{"subject": {{"value": "...", "type": "..."}}, "predicate": {{"value": "...", "type": "Relationship"}}, "object": {{"value": "...", "type": "..."}}}}]}}
""")

            conversation.append(final_prompt)

            # Log final prompt
            if self.logger:
                self.logger.log_llm_call(
                    [{"role": "user", "content": final_prompt.content}],
                    "",
                    {"model": self.llm.model_name, "provider": "Groq"}
                )

            # Genera triplette con structured output
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(conversation)

            # Filtra e converti (matrice 2x3)
            valid_triplets = response.filter_valid()
            iot_triplets = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid_triplets
            ]

            # Log risultato
            if self.logger:
                filtered_count = len(response.triplets) - len(valid_triplets)
                if filtered_count > 0:
                    self.logger.console.print(f"[dim]Filtered out {filtered_count} invalid triplets[/dim]")
                self.logger.log_agent_response(f"Generated {len(iot_triplets)} IoT triplets:\n{json.dumps(iot_triplets, indent=2)}")

            # IMPORTANTE: operator.add funziona solo per nodi paralleli, non sequenziali!
            # Dobbiamo manualmente combinare con le augmented_triplets esistenti
            existing_augmented = augmented_triplets  # Già lette all'inizio del nodo
            combined = existing_augmented + iot_triplets

            if self.logger:
                self.logger.console.print(f"[blue]📊 Combining triplets:[/blue]")
                self.logger.console.print(f"[blue]  - Existing augmented (text): {len(existing_augmented)}[/blue]")
                self.logger.console.print(f"[blue]  - New IoT: {len(iot_triplets)}[/blue]")
                self.logger.console.print(f"[blue]  - Total combined: {len(combined)}[/blue]")

            return {"augmented_triplets": combined}

        except Exception as e:
            error_msg = f"Errore IoT triplet generation: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "IoT Generation Node")
            return {
                "error": error_msg,
                "augmented_triplets": []
            }

    def _validation_decide_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 5a: Decide se eseguire validazione guardrail.

        Analizza tutte le triplette finali (estratte + augmented) e decide
        se serve validazione per consistenza e qualità.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con validation_should_run
        """
        extracted = state.get("triplets", [])
        augmented = state.get("augmented_triplets", [])
        all_triplets = extracted + augmented

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold blue]═══ Validation Decision Node ═══[/bold blue]")
            self.logger.console.print(f"[dim]Analyzing {len(all_triplets)} total triplets:[/dim]")
            self.logger.console.print(f"[dim]  - Extracted: {len(extracted)}[/dim]")
            self.logger.console.print(f"[dim]  - Augmented (text + IoT): {len(augmented)}[/dim]")
            self.logger.console.print(f"[dim]  - Total: {len(all_triplets)}[/dim]")

        # Se poche triplette, skippa validazione
        if len(all_triplets) < 5:
            if self.logger:
                self.logger.console.print(f"[yellow]Too few triplets ({len(all_triplets)}), skipping validation[/yellow]")
            return {
                "validation_should_run": False,
                "validation_iteration": 0,
                "validated_triplets": all_triplets
            }

        # Prepara prompt
        input_text = state.get("input_text", "")
        triplets_str = "\n".join([
            f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for t in all_triplets
        ])

        messages = self.prompt_manager.build_messages(
            'validation_decide',
            original_text=input_text[:500],  # Prime 500 char come context
            triplets=triplets_str,
            count=len(all_triplets)
        )

        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            response = self.llm.invoke(lc_messages)
            decision = response.content.strip().upper()
            should_validate = any(kw in decision for kw in ["YES", "SI", "VALIDATE", "NEEDED", "INCONSISTEN"])

            if self.logger:
                emoji = "✅" if should_validate else "❌"
                self.logger.log_agent_response(f"{emoji} Decision: {'RUN validation' if should_validate else 'SKIP validation'}\n\n{response.content}")

            return {
                "validation_should_run": should_validate,
                "validation_iteration": 0,
                "validated_triplets": all_triplets if not should_validate else [],
                "validation_reasoning": [f"Decision: {'Validate' if should_validate else 'Skip'}"]
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Errore validation decision: {str(e)}", "Validation Decision")
            return {
                "validation_should_run": False,
                "validation_iteration": 0,
                "validated_triplets": all_triplets
            }

    def _validation_iterate_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 5b: Esegue UNA iterazione di validazione guardrail.

        Il Guardrail Agent:
        1. Riceve testo originale + triplette correnti
        2. Identifica inconsistenze, ridondanze, errori semantici
        3. Rimuove/corregge triplette problematiche
        4. Decide se continuare validazione o finire

        Self-loop: può iterare fino a 3 volte per raffinare progressivamente.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update con validated_triplets, removed_triplets, validation_iteration
        """
        iteration = state.get("validation_iteration", 0)
        current_triplets = state.get("validated_triplets", [])
        if not current_triplets:
            # Prima iterazione: prendi tutte
            current_triplets = state.get("triplets", []) + state.get("augmented_triplets", [])

        input_text = state.get("input_text", "")

        # Log
        if self.logger:
            self.logger.console.print(f"\n[bold magenta]🛡️  Validation Iteration {iteration + 1}/3[/bold magenta]")
            self.logger.console.print(f"[dim]Validating {len(current_triplets)} triplets against original text[/dim]")

        triplets_str = "\n".join([
            f"{i+1}. {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for i, t in enumerate(current_triplets)
        ])

        messages = self.prompt_manager.build_messages(
            'validation_iterate',
            original_text=input_text,
            triplets=triplets_str,
            iteration=iteration + 1
        )

        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        if self.logger and iteration == 0:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            import json

            # Structured output con Pydantic
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(lc_messages)

            # Filtra triplette valide (matrice 2x3)
            valid = response.filter_valid()
            validated = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid
            ]

            # Identifica rimosse
            removed_count = len(current_triplets) - len(validated)

            # Reasoning: decide se continuare
            should_continue = removed_count > 0 and iteration < 2  # Max 3 iterazioni

            # Log
            if self.logger:
                from rich.panel import Panel
                if removed_count > 0:
                    self.logger.console.print(f"[yellow]🗑️  Removed {removed_count} inconsistent triplets[/yellow]")
                else:
                    self.logger.console.print(f"[green]✅ All triplets validated, no changes needed[/green]")

                decision = "CONTINUE validation" if should_continue else "FINISH validation"
                self.logger.log_agent_response(f"Iteration {iteration + 1}: Validated {len(validated)}/{len(current_triplets)} triplets\n\nDecision: {decision}")

            return {
                "validated_triplets": validated,
                "validation_iteration": iteration + 1,
                "validation_reasoning": [f"Iteration {iteration + 1}: {removed_count} removed"],
                "validation_should_run": should_continue
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Errore validation iteration: {str(e)}", "Validation Iterate")
            # In caso di errore, passa tutte le triplette senza modifiche
            return {
                "validated_triplets": current_triplets,
                "validation_iteration": iteration + 1,
                "validation_should_run": False
            }

    def _should_use_iot(self, state: GraphState) -> str:
        """
        Conditional edge: decide se iniziare l'esplorazione IoT.

        Args:
            state: Stato corrente del grafo

        Returns:
            "explore" se l'agent ha deciso di usare IoT, "skip" altrimenti
        """
        should_explore = state.get("iot_should_explore", False)

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_use_iot[/bold cyan]")
            self.logger.console.print(f"[dim]Decision: {'EXPLORE' if should_explore else 'SKIP'}[/dim]")

        if should_explore:
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: iot_react (start ReAct loop)[/green]")
            return "explore"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: finalize (skip IoT)[/yellow]")
            return "skip"

    def _should_continue_react(self, state: GraphState) -> str:
        """
        Conditional edge: decide se continuare il ReAct loop.

        Controlla:
        - Se l'agent ha deciso di continuare (iot_should_explore)
        - Se non abbiamo raggiunto max_iterations (5)

        Args:
            state: Stato corrente del grafo

        Returns:
            "continue" per continuare il loop, "finish" per generare triplette
        """
        iteration = state.get("iot_react_iteration", 0)
        should_explore = state.get("iot_should_explore", True)
        max_iterations = 5

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_react[/bold cyan]")
            self.logger.console.print(f"[dim]Iteration: {iteration}/{max_iterations}, Should explore: {should_explore}[/dim]")

        # Continua se: ancora vuole esplorare AND non ha raggiunto max iterations
        if should_explore and iteration < max_iterations:
            if self.logger:
                self.logger.console.print(f"[green]➡️  CONTINUE: More data needed[/green]")
            return "continue"
        else:
            reason = "max iterations reached" if iteration >= max_iterations else "agent decided to stop"
            if self.logger:
                self.logger.console.print(f"[yellow]✅ FINISH: {reason}[/yellow]")
            return "finish"

    def _should_validate(self, state: GraphState) -> str:
        """
        Conditional edge: decide se iniziare validazione guardrail.

        Args:
            state: Stato corrente del grafo

        Returns:
            "validate" per avviare validazione, "skip" per saltare
        """
        should_validate = state.get("validation_should_run", False)

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_validate[/bold cyan]")
            self.logger.console.print(f"[dim]Decision: {'VALIDATE' if should_validate else 'SKIP'}[/dim]")

        if should_validate:
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: validation_iterate (start validation loop)[/green]")
            return "validate"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: finalize (skip validation)[/yellow]")
            return "skip"

    def _should_continue_validation(self, state: GraphState) -> str:
        """
        Conditional edge: decide se continuare il loop di validazione.

        Controlla:
        - Se l'agent ha deciso di continuare (validation_should_run)
        - Se non abbiamo raggiunto max_iterations (3)

        Args:
            state: Stato corrente del grafo

        Returns:
            "continue" per iterare ancora, "finish" per terminare
        """
        iteration = state.get("validation_iteration", 0)
        should_continue = state.get("validation_should_run", False)
        max_iterations = 3

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_validation[/bold cyan]")
            self.logger.console.print(f"[dim]Iteration: {iteration}/{max_iterations}, Should continue: {should_continue}[/dim]")

        if should_continue and iteration < max_iterations:
            if self.logger:
                self.logger.console.print(f"[green]➡️  CONTINUE: More validation needed[/green]")
            return "continue"
        else:
            reason = "max iterations reached" if iteration >= max_iterations else "validation complete"
            if self.logger:
                self.logger.console.print(f"[yellow]✅ FINISH: {reason}[/yellow]")
            return "finish"

    def _finalize_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 6: Finalizza con le triplette validate (se la validazione è stata eseguita).

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con le triplette finali
        """
        # Se validazione eseguita, usa validated_triplets; altrimenti usa extracted + augmented
        validated = state.get("validated_triplets", [])

        if validated:
            final = validated
            validation_run = True
        else:
            extracted = state.get("triplets", [])
            augmented = state.get("augmented_triplets", [])
            final = extracted + augmented
            validation_run = False

        # Log finale
        if self.logger:
            self.logger.console.print(f"\n[bold white]═══ Finalize Node ═══[/bold white]")
            self.logger.console.print(f"[green]Final triplets: {len(final)}[/green]")

            if validation_run:
                validation_iterations = state.get("validation_iteration", 0)
                self.logger.console.print(f"[blue]  ✓ Validated through {validation_iterations} guardrail iteration(s)[/blue]")
            else:
                extracted = state.get("triplets", [])
                augmented = state.get("augmented_triplets", [])
                self.logger.console.print(f"  - Extracted from text: {len(extracted)}")
                self.logger.console.print(f"  - Augmented (text + IoT): {len(augmented)}")

            self.logger.log_summary()

        return {"final_triplets": final}

    def run(
        self,
        input_text: str,
        chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Esegue il grafo di estrazione triplette.

        Args:
            input_text: Testo da cui estrarre triplette
            chunk_size: Dimensione dei chunk in caratteri

        Returns:
            Risultato finale con tutte le triplette estratte e augmented
        """
        # Inizializza lo stato
        initial_state = {
            # Text processing
            "input_text": input_text,
            "chunk_size": chunk_size,
            "chunks": [],
            "current_chunk_index": 0,
            "previous_summary": None,

            # Triplets
            "triplets": [],
            "augmented_triplets": [],
            "final_triplets": [],

            # IoT ReAct Loop
            "iot_should_explore": False,
            "iot_react_iteration": 0,
            "iot_data_collected": {},
            "iot_reasoning_history": [],
            "iot_conversation": [],

            # Validation Guardrail
            "validation_should_run": False,
            "validation_iteration": 0,
            "validation_reasoning": [],
            "validated_triplets": [],
            "removed_triplets": [],

            # Error handling
            "error": None
        }

        # Esegui il grafo
        result = self.graph.invoke(initial_state)

        return result
