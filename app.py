"""
Streamlit Frontend per Human Digital Twin
"""

import streamlit as st
import sys
from pathlib import Path
import json
import requests
from datetime import datetime

# Aggiungi il path del progetto al PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.llm import LLMFactory
from src.prompts import PromptManager
from src.agents import MCPAgent
from src.data_generator import OntologyDataGenerator


# Inizializzazione della configurazione
@st.cache_resource
def init_config():
    """Inizializza la configurazione."""
    return ConfigManager()


@st.cache_resource
def init_prompt_manager():
    """Inizializza il PromptManager."""
    return PromptManager()


@st.cache_resource
def init_llm(_config):
    """Inizializza il servizio LLM."""
    llm_config = _config.get_llm_config()
    provider = llm_config.get('provider', 'groq')
    api_key = _config.get_env('GROQ_API_KEY')

    if not api_key:
        st.error("GROQ_API_KEY non trovata nel file .env")
        st.stop()

    return LLMFactory.create(provider, llm_config, api_key)


@st.cache_resource
def init_triplet_graph(_config, _llm, _graph_version="v5_cascade_text_iot_aug"):
    """Inizializza il grafo LangGraph per estrazione triplette."""
    from src.agents.triplet_extraction_graph import TripletExtractionGraph

    api_key = _config.get_env('GROQ_API_KEY')
    mcp_config = _config.get_mcp_config()
    mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

    return TripletExtractionGraph(
        llm_api_key=api_key,
        llm_model=_llm.model,
        mcp_base_url=mcp_url,
        enable_logging=True  # Abilita logging Rich
    )


def main():
    """Main function per l'app Streamlit."""

    # Configurazione della pagina
    config = init_config()
    streamlit_config = config.get_streamlit_config()

    st.set_page_config(
        page_title=streamlit_config.get('title', 'Human Digital Twin'),
        layout="wide"
    )

    st.title(streamlit_config.get('title', 'Human Digital Twin'))

    # Inizializza i componenti
    prompt_manager = init_prompt_manager()
    llm = init_llm(config)
    triplet_graph = init_triplet_graph(config, llm)

    # Sidebar per la configurazione
    with st.sidebar:
        st.header("Configurazione")

        # Info sul modello
        model_info = llm.get_model_info()
        st.write(f"**Provider:** {model_info['provider']}")
        st.write(f"**Modello:** {model_info['model']}")

        # Configurazione MCP
        mcp_config = config.get_mcp_config()
        mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"
        st.write(f"**MCP Server:** {mcp_url}")

    # Tabs per le diverse funzionalità
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Estrazione Triplette",
        "Ontology Validation",
        "Knowledge Graph Builder",
        "Dati IoT",
        "Servizi Esterni",
        "Chat Agent"
    ])

    # Tab 1: Estrazione Triplette da Testo
    with tab1:
        st.header("Estrazione Triplette da Testo con LangGraph")
        st.write("Estrazione multi-stage di triplette RDF con augmentation dati IoT")

        # Scelta input: textbox o file JSON
        input_mode = st.radio(
            "Sorgente del testo:",
            ["Textbox", "File JSON"],
            horizontal=True
        )

        text_input = None

        if input_mode == "Textbox":
            text_input = st.text_area(
                "Testo da analizzare:",
                height=200,
                placeholder="Inserisci qui il testo..."
            )
        else:
            uploaded_file = st.file_uploader("Carica file JSON", type=['json'])
            if uploaded_file:
                try:
                    json_data = json.load(uploaded_file)
                    # Assume che il JSON abbia un campo "text" o simile
                    text_input = json_data.get("text", json.dumps(json_data))
                    st.text_area("Testo estratto:", text_input, height=150, disabled=True)
                except Exception as e:
                    st.error(f"Errore nel parsing del JSON: {str(e)}")

        # Slider per chunk size
        chunk_size = st.slider(
            "Dimensione chunk (caratteri):",
            min_value=200,
            max_value=3000,
            value=1000,
            step=100
        )

        if st.button("Estrai Triplette con LangGraph", key="extract_triplets"):
            if text_input:
                with st.spinner("Estrazione multi-stage in corso..."):
                    try:
                        # Esegui il grafo (recupera automaticamente i dati IoT dal MCP)
                        result = triplet_graph.run(
                            input_text=text_input,
                            chunk_size=chunk_size
                        )

                        # Mostra risultati
                        if result.get("error"):
                            st.error(f"Errore: {result['error']}")
                        else:
                            st.success("Estrazione completata!")

                            # Triplette estratte
                            triplets = result.get("triplets", [])
                            st.subheader(f"Triplette estratte: {len(triplets)}")
                            if triplets:
                                # Visualizzazione tabellare con tipi (matrice 2x3)
                                for idx, t in enumerate(triplets[:10], 1):  # Mostra prime 10
                                    with st.expander(f"Tripletta #{idx}: {t.get('subject', {}).get('value', '')} → {t.get('predicate', {}).get('value', '')} → {t.get('object', {}).get('value', '')}"):
                                        # Riga 1: Values
                                        st.markdown("**Valori (Instance):**")
                                        st.code(f"{t.get('subject', {}).get('value', '')}  →  {t.get('predicate', {}).get('value', '')}  →  {t.get('object', {}).get('value', '')}")

                                        # Riga 2: Types
                                        st.markdown("**Tipi (Class):**")
                                        st.code(f"{t.get('subject', {}).get('type', 'N/A')}  →  {t.get('predicate', {}).get('type', 'N/A')}  →  {t.get('object', {}).get('type', 'N/A')}")

                                if len(triplets) > 10:
                                    st.info(f"Mostrate prime 10 triplette. Totale: {len(triplets)}")

                                # JSON completo in expander
                                with st.expander("🔍 Visualizza JSON completo"):
                                    st.json(triplets)

                            # Triplette augmented
                            augmented = result.get("augmented_triplets", [])
                            if augmented:
                                st.subheader(f"Triplette augmented (text + IoT): {len(augmented)}")
                                for idx, t in enumerate(augmented[:10], 1):
                                    with st.expander(f"Augmented #{idx}: {t.get('subject', {}).get('value', '')} → {t.get('predicate', {}).get('value', '')} → {t.get('object', {}).get('value', '')}"):
                                        st.markdown("**Valori:**")
                                        st.code(f"{t.get('subject', {}).get('value', '')}  →  {t.get('predicate', {}).get('value', '')}  →  {t.get('object', {}).get('value', '')}")
                                        st.markdown("**Tipi:**")
                                        st.code(f"{t.get('subject', {}).get('type', 'N/A')}  →  {t.get('predicate', {}).get('type', 'N/A')}  →  {t.get('object', {}).get('type', 'N/A')}")

                                if len(augmented) > 10:
                                    st.info(f"Mostrate prime 10 augmented. Totale: {len(augmented)}")

                                with st.expander("🔍 Visualizza JSON completo"):
                                    st.json(augmented)

                            # Triplette finali
                            final = result.get("final_triplets", [])
                            st.subheader(f"✅ Totale triplette finali: {len(final)}")

                            # Salva in session_state per riuso nella tab Ontology
                            st.session_state['extracted_triplets'] = final
                            st.session_state['extraction_result'] = result

                            # Auto-save session se configurato
                            sessions_config = config.get('sessions', {})
                            if sessions_config.get('auto_save', True):
                                from src.utils import SessionManager

                                session_mgr = SessionManager(sessions_config.get('sessions_dir', 'data/sessions'))

                                # Prepara metadata
                                metadata = {
                                    'input_text_preview': text_input[:200] + "..." if len(text_input) > 200 else text_input,
                                    'chunk_size': chunk_size,
                                    'total_chunks': len(result.get('chunks', [])),
                                    'extracted_count': len(result.get('triplets', [])),
                                    'augmented_count': len(result.get('augmented_triplets', []))
                                }

                                saved_path = session_mgr.save_session(final, metadata)
                                st.session_state['last_saved_session'] = saved_path
                                st.success(f"💾 Sessione salvata automaticamente: `{Path(saved_path).name}`")

                            # Download JSON
                            json_output = json.dumps(final, indent=2)
                            st.download_button(
                                label="Download triplette (JSON)",
                                data=json_output,
                                file_name="triplets.json",
                                mime="application/json"
                            )

                            # Pulsante per andare alla validazione
                            st.info("💡 Triplette salvate! Vai alla tab **Ontology Validation** per validarle con Schema.org")

                    except Exception as e:
                        st.error(f"Errore durante l'estrazione: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Inserisci del testo o carica un file JSON prima di procedere")

    # Tab 2: Ontology Validation
    with tab2:
        st.header("🔍 Ontology Validation con Schema.org")
        st.write("Valida le triplette estratte confrontandole semanticamente con l'ontologia Schema.org")

        # UI per caricamento sessioni
        st.subheader("📂 Carica Sessione")

        col1, col2 = st.columns([3, 1])

        with col1:
            load_mode = st.radio(
                "Sorgente triplette:",
                ["Sessione corrente (in memoria)", "Carica da sessione salvata", "Carica file JSON custom"],
                horizontal=False
            )

        # Gestisci caricamento da sessione salvata
        if load_mode == "Carica da sessione salvata":
            from src.utils import SessionManager

            sessions_config = config.get('sessions', {})
            session_mgr = SessionManager(sessions_config.get('sessions_dir', 'data/sessions'))

            # Lista sessioni disponibili
            sessions = session_mgr.list_sessions()

            if not sessions:
                st.warning("⚠️ Nessuna sessione salvata trovata. Estrai triplette prima nella tab **Estrazione Triplette**.")
            else:
                st.info(f"📦 Trovate {len(sessions)} sessioni salvate")

                # Selectbox per scegliere la sessione
                session_options = [
                    f"{s['filename']} - {s['triplets_count']} triplette ({s['timestamp'][:19]})"
                    for s in sessions
                ]

                selected_idx = st.selectbox(
                    "Seleziona sessione da caricare:",
                    range(len(sessions)),
                    format_func=lambda i: session_options[i]
                )

                selected_session = sessions[selected_idx]

                # Mostra preview metadata
                with st.expander("🔍 Info Sessione"):
                    st.write(f"**File:** {selected_session['filename']}")
                    st.write(f"**Timestamp:** {selected_session['timestamp']}")
                    st.write(f"**Triplette:** {selected_session['triplets_count']}")

                    metadata = selected_session.get('metadata', {})
                    if metadata:
                        st.write("**Metadata:**")
                        st.json(metadata)

                # Pulsante caricamento
                if st.button("📥 Carica Sessione", type="primary"):
                    session_data = session_mgr.load_session(selected_session['filepath'])
                    st.session_state['extracted_triplets'] = session_data['triplets']
                    st.session_state['loaded_from_file'] = selected_session['filename']
                    st.success(f"✅ Sessione caricata: {selected_session['filename']}")
                    st.rerun()

        # Gestisci caricamento da file JSON custom
        elif load_mode == "Carica file JSON custom":
            uploaded_file = st.file_uploader("Carica file JSON con triplette", type=['json'])

            if uploaded_file:
                try:
                    loaded_data = json.load(uploaded_file)

                    # Supporta diversi formati
                    if 'triplets' in loaded_data:
                        triplets = loaded_data['triplets']
                    elif isinstance(loaded_data, list):
                        triplets = loaded_data
                    else:
                        st.error("❌ Formato JSON non riconosciuto. Atteso: lista di triplette o oggetto con campo 'triplets'")
                        triplets = []

                    if triplets:
                        st.success(f"✅ Caricato file con {len(triplets)} triplette")

                        # Mostra preview
                        with st.expander("🔍 Preview Triplette"):
                            st.json(triplets[:3])

                        # Pulsante per usare queste triplette
                        if st.button("📥 Usa Queste Triplette", type="primary"):
                            st.session_state['extracted_triplets'] = triplets
                            st.session_state['loaded_from_file'] = uploaded_file.name
                            st.success(f"✅ Triplette caricate da: {uploaded_file.name}")
                            st.rerun()

                except json.JSONDecodeError as e:
                    st.error(f"❌ Errore nel parsing JSON: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Errore: {str(e)}")

        # Controlla se ci sono triplette da validare
        if 'extracted_triplets' not in st.session_state or not st.session_state['extracted_triplets']:
            if load_mode == "Sessione corrente (in memoria)":
                st.warning("⚠️ Nessuna tripletta in memoria. Vai alla tab **Estrazione Triplette** o carica una sessione salvata.")
        else:
            triplets_to_validate = st.session_state['extracted_triplets']

            # Mostra info fonte
            source_info = ""
            if 'loaded_from_file' in st.session_state:
                source_info = f" (caricato da: `{st.session_state['loaded_from_file']}`)"

            st.success(f"✅ Trovate **{len(triplets_to_validate)}** triplette da validare{source_info}")

            # Sidebar per configurazione
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🛠️ Validazione Config")

                # Soglia di confidence
                ontology_config = config.get_ontology_config()
                validation_threshold = st.slider(
                    "Soglia confidence minima:",
                    min_value=0.0,
                    max_value=1.0,
                    value=ontology_config.get('validation_threshold', 0.5),
                    step=0.05,
                    help="Triplette con score sotto questa soglia saranno marcate come low-confidence"
                )

                # Rate limit
                rate_limit = st.slider(
                    "Rate limit (sec):",
                    min_value=0.0,
                    max_value=10.0,
                    value=ontology_config.get('rate_limit_delay', 2.0),
                    step=0.5,
                    help="Pausa tra richieste API (2.0+ raccomandato per Cohere trial, evita rate limit 429)"
                )

            # Inizializza servizi ontology (cached)
            @st.cache_resource
            def init_ontology_services(_config):
                """Inizializza i servizi per validazione ontology."""
                from src.ontology.schema_downloader import ensure_schema_org
                from src.ontology import SchemaOrgLoader, EmbeddingService

                # Download schema.jsonld se necessario
                with st.spinner("📥 Verifico disponibilità Schema.org ontology..."):
                    schema_path = ensure_schema_org(_config)
                    if not schema_path:
                        st.error("❌ Impossibile scaricare Schema.org ontology")
                        return None, None

                # Carica ontologia
                with st.spinner("📚 Caricamento Schema.org ontology..."):
                    ontology = SchemaOrgLoader(schema_path)
                    st.success(f"✅ Ontologia caricata: {len(ontology.get_all_classes())} classi, {len(ontology.get_all_properties())} proprietà")

                # Inizializza embedding service
                ontology_config = _config.get_ontology_config()
                provider = ontology_config.get('embedding_provider', 'cohere')
                cache_dir = ontology_config.get('cache_dir', 'data/ontology/cache')

                api_key = None
                if provider == 'cohere':
                    api_key = _config.get_env('COHERE_API_KEY')
                elif provider == 'mistral':
                    api_key = _config.get_env('MISTRAL_API_KEY')

                if not api_key and provider in ['cohere', 'mistral']:
                    st.error(f"❌ {provider.upper()}_API_KEY non trovata nel file .env")
                    return None, None

                embeddings = EmbeddingService(
                    provider=provider,
                    api_key=api_key,
                    use_cache=True,
                    cache_dir=cache_dir
                )

                return ontology, embeddings

            ontology, embeddings = init_ontology_services(config)

            if ontology and embeddings:
                # Pulsante per pre-calcolare embeddings
                cache_size = len(embeddings.cache.cache) if embeddings.cache else 0
                total_items = len(ontology.get_all_classes()) + len(ontology.get_all_properties())

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.metric("Embeddings in cache", f"{cache_size}/{total_items}")
                with col2:
                    completion_pct = (cache_size / total_items * 100) if total_items > 0 else 0
                    st.metric("Completamento", f"{completion_pct:.1f}%")

                # Pre-compute embeddings se necessario
                if cache_size < total_items:
                    if st.button("🚀 Pre-calcola tutti gli embeddings", help="Calcola gli embeddings per tutte le classi/proprietà (necessario solo la prima volta)"):
                        with st.spinner("⏳ Pre-calcolo embeddings in corso..."):
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()

                            # Raccogli tutti i testi da embeddare
                            # IMPORTANTE: Usa la STESSA logica di TripleMatcher per evitare cache miss!
                            all_texts = []

                            # Classes con enriched context (come TripleMatcher._get_class_embedding)
                            for class_name in ontology.get_all_classes():
                                class_desc = ontology.get_class_description(class_name)
                                class_info = ontology.get_class_info(class_name)

                                parts = [f"{class_name}"]
                                if class_desc:
                                    parts.append(class_desc)

                                # Add parent classes for context
                                parent_classes = class_info.get('subClassOf', [])
                                if parent_classes:
                                    parents_str = ", ".join(parent_classes[:3])
                                    parts.append(f"Type of: {parents_str}")

                                text = ". ".join(parts)
                                all_texts.append(text)

                            # Properties con enriched context (come TripleMatcher._get_property_embedding)
                            for prop_name in ontology.get_all_properties():
                                prop_desc = ontology.get_property_description(prop_name)
                                prop_info = ontology.get_property_info(prop_name)

                                parts = [f"{prop_name}"]
                                if prop_desc:
                                    parts.append(prop_desc)

                                # Add domain context
                                domain_classes = prop_info.get('domainIncludes', [])
                                if domain_classes:
                                    domain_str = ", ".join(domain_classes[:3])
                                    parts.append(f"Used with: {domain_str}")

                                # Add range context
                                range_classes = prop_info.get('rangeIncludes', [])
                                if range_classes:
                                    range_str = ", ".join(range_classes[:3])
                                    parts.append(f"Points to: {range_str}")

                                text = ". ".join(parts)
                                all_texts.append(text)

                            # Processa a batch
                            batch_size = 20
                            for i in range(0, len(all_texts), batch_size):
                                batch = all_texts[i:i + batch_size]

                                try:
                                    embeddings.embed_texts(batch, input_type="search_document", rate_limit_delay=rate_limit)
                                    embeddings.cache.save_cache()

                                    progress = min(1.0, (i + batch_size) / len(all_texts))
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processati {min(i + batch_size, len(all_texts))}/{len(all_texts)} embeddings...")

                                except Exception as e:
                                    st.error(f"❌ Errore durante pre-calcolo: {str(e)}")
                                    break

                            st.success("✅ Pre-calcolo completato!")
                            st.rerun()

                # Pulsante per validare triplette
                st.markdown("---")
                if st.button("🔍 Valida Triplette con Schema.org", type="primary"):
                    from src.ontology import TripleMatcher

                    matcher = TripleMatcher(ontology, embeddings, rate_limit_delay=rate_limit)

                    validated_results = []

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    for idx, triplet in enumerate(triplets_to_validate):
                        status_text.text(f"Validazione tripletta {idx + 1}/{len(triplets_to_validate)}...")

                        # Estrai valori e tipi dalla matrice 2x3
                        subject_data = triplet.get('subject', {})
                        predicate_data = triplet.get('predicate', {})
                        obj_data = triplet.get('object', {})

                        subject_value = subject_data.get('value', '') if isinstance(subject_data, dict) else subject_data
                        predicate_value = predicate_data.get('value', '') if isinstance(predicate_data, dict) else predicate_data
                        obj_value = obj_data.get('value', '') if isinstance(obj_data, dict) else obj_data

                        subject_type = subject_data.get('type', None) if isinstance(subject_data, dict) else None
                        predicate_type = predicate_data.get('type', None) if isinstance(predicate_data, dict) else None
                        obj_type = obj_data.get('type', None) if isinstance(obj_data, dict) else None

                        if subject_value and predicate_value and obj_value:
                            result = matcher.match_triple(
                                subject_value, predicate_value, obj_value,
                                subject_type, predicate_type, obj_type
                            )
                            validated_results.append(result)

                        progress_bar.progress((idx + 1) / len(triplets_to_validate))

                    status_text.empty()
                    progress_bar.empty()

                    # Salva risultati in session_state
                    st.session_state['validation_results'] = validated_results
                    st.session_state['validation_threshold'] = validation_threshold

                    st.success(f"✅ Validazione completata per {len(validated_results)} triplette!")
                    st.rerun()

                # Mostra risultati validazione se disponibili
                if 'validation_results' in st.session_state and st.session_state['validation_results']:
                    st.markdown("---")
                    st.subheader("📊 Risultati Validazione")

                    results = st.session_state['validation_results']
                    threshold = st.session_state.get('validation_threshold', 0.5)

                    # Filtra per soglia
                    valid_results = [r for r in results if r.get('mu', 0.0) >= threshold]
                    low_conf_results = [r for r in results if r.get('mu', 0.0) < threshold]

                    # Statistiche
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Valid", len(valid_results))
                    with col2:
                        st.metric("⚠️ Low Confidence", len(low_conf_results))
                    with col3:
                        avg_score = sum(r.get('mu', 0.0) for r in results) / len(results) if results else 0
                        st.metric("📈 Score Medio", f"{avg_score:.3f}")

                    # Distribuzione score (istogramma)
                    import plotly.express as px
                    scores = [r.get('mu', 0.0) for r in results]
                    fig = px.histogram(
                        x=scores,
                        nbins=20,
                        title="Distribuzione Score di Confidenza",
                        labels={'x': 'Score μ', 'y': 'Frequenza'}
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Soglia")
                    st.plotly_chart(fig, use_container_width=True)

                    # Tab per risultati validi e low-confidence
                    result_tab1, result_tab2 = st.tabs(["✅ Validated Triplets", "⚠️ Low Confidence Triplets"])

                    with result_tab1:
                        if valid_results:
                            for idx, result in enumerate(valid_results, 1):
                                with st.expander(f"Tripletta #{idx} - Score: {result.get('mu', 0.0):.3f}"):
                                    # Tripletta originale
                                    st.markdown(f"**Tripletta:** `{result['subject']['value']}` → `{result['predicate']['value']}` → `{result['object']['value']}`")

                                    # Matching info (matrice 2x3: LLM types + Schema.org types)
                                    col_s, col_p, col_o = st.columns(3)

                                    with col_s:
                                        st.markdown("**🎯 Subject**")
                                        orig_type = result['subject'].get('original_type')
                                        if orig_type:
                                            st.markdown(f"LLM Type: `{orig_type}`")
                                        st.markdown(f"Schema.org: `{result['subject']['matched_class']}`")
                                        st.markdown(f"Score: `{result['subject']['confidence']:.3f}`")

                                    with col_p:
                                        st.markdown("**🔗 Predicate**")
                                        orig_type = result['predicate'].get('original_type')
                                        if orig_type:
                                            st.markdown(f"LLM Type: `{orig_type}`")
                                        st.markdown(f"Schema.org: `{result['predicate']['matched_property']}`")
                                        st.markdown(f"Score: `{result['predicate']['confidence']:.3f}`")

                                    with col_o:
                                        st.markdown("**📍 Object**")
                                        orig_type = result['object'].get('original_type')
                                        if orig_type:
                                            st.markdown(f"LLM Type: `{orig_type}`")
                                        st.markdown(f"Schema.org: `{result['object']['matched_class']}`")
                                        st.markdown(f"Score: `{result['object']['confidence']:.3f}`")

                                    # Branch path
                                    st.markdown(f"**🌳 Path:** {result.get('branch_path', 'N/A')}")

                                    # Branch strategies comparison (matrice 3 strategie)
                                    with st.expander("🌳 Branch Strategies (3 metodi di esplorazione)"):
                                        all_branches = result.get('all_branches', [])
                                        if all_branches:
                                            # Raggruppa per method_used
                                            predicate_driven = [b for b in all_branches if b.get('method_used') == 'predicate_driven']
                                            subject_driven = [b for b in all_branches if b.get('method_used') == 'subject_driven']
                                            object_driven = [b for b in all_branches if b.get('method_used') == 'object_driven']

                                            col1, col2, col3 = st.columns(3)

                                            with col1:
                                                st.markdown("**🔗 Predicate-Driven**")
                                                st.caption("Predicate → Subject/Object")
                                                if predicate_driven:
                                                    best = predicate_driven[0]
                                                    st.metric("Score μ", f"{best['mu']:.3f}")
                                                    st.caption(f"Path: {best.get('branch_path', 'N/A')}")
                                                else:
                                                    st.caption("Nessun match trovato")

                                            with col2:
                                                st.markdown("**🎯 Subject-Driven**")
                                                st.caption("Subject → Predicate → Object")
                                                if subject_driven:
                                                    best = subject_driven[0]
                                                    st.metric("Score μ", f"{best['mu']:.3f}")
                                                    st.caption(f"Path: {best.get('branch_path', 'N/A')}")
                                                else:
                                                    st.caption("Nessun match trovato")

                                            with col3:
                                                st.markdown("**📍 Object-Driven**")
                                                st.caption("Object → Predicate → Subject")
                                                if object_driven:
                                                    best = object_driven[0]
                                                    st.metric("Score μ", f"{best['mu']:.3f}")
                                                    st.caption(f"Path: {best.get('branch_path', 'N/A')}")
                                                else:
                                                    st.caption("Nessun match trovato")

                                            st.markdown("---")
                                            st.markdown(f"**✅ Strategia scelta:** `{result.get('method_used', 'N/A')}` (score più alto)")

                                    # Top candidates (collapsible)
                                    with st.expander("🔍 Top Candidates"):
                                        st.markdown("**Subject candidates:**")
                                        for name, score in result['subject']['top_candidates'][:3]:
                                            st.markdown(f"- {name}: {score:.3f}")

                                        st.markdown("**Predicate candidates:**")
                                        for name, score in result['predicate']['top_candidates'][:3]:
                                            st.markdown(f"- {name}: {score:.3f}")

                                        st.markdown("**Object candidates:**")
                                        for name, score in result['object']['top_candidates'][:3]:
                                            st.markdown(f"- {name}: {score:.3f}")
                        else:
                            st.info("Nessuna tripletta sopra la soglia")

                    with result_tab2:
                        if low_conf_results:
                            st.warning(f"⚠️ {len(low_conf_results)} triplette sotto la soglia {threshold}")

                            for idx, result in enumerate(low_conf_results, 1):
                                with st.expander(f"Tripletta #{idx} - Score: {result.get('mu', 0.0):.3f}"):
                                    # Tripletta originale
                                    st.markdown(f"**Tripletta:** `{result['subject']['value']}` → `{result['predicate']['value']}` → `{result['object']['value']}`")

                                    # Matching info (matrice 2x3: LLM types + Schema.org types)
                                    col_s, col_p, col_o = st.columns(3)

                                    with col_s:
                                        st.markdown("**🎯 Subject**")
                                        orig_type = result['subject'].get('original_type')
                                        if orig_type:
                                            st.markdown(f"LLM Type: `{orig_type}`")
                                        st.markdown(f"Schema.org: `{result['subject']['matched_class']}`")
                                        st.markdown(f"Score: `{result['subject']['confidence']:.3f}`")

                                    with col_p:
                                        st.markdown("**🔗 Predicate**")
                                        orig_type = result['predicate'].get('original_type')
                                        if orig_type:
                                            st.markdown(f"LLM Type: `{orig_type}`")
                                        st.markdown(f"Schema.org: `{result['predicate']['matched_property']}`")
                                        st.markdown(f"Score: `{result['predicate']['confidence']:.3f}`")

                                    with col_o:
                                        st.markdown("**📍 Object**")
                                        orig_type = result['object'].get('original_type')
                                        if orig_type:
                                            st.markdown(f"LLM Type: `{orig_type}`")
                                        st.markdown(f"Schema.org: `{result['object']['matched_class']}`")
                                        st.markdown(f"Score: `{result['object']['confidence']:.3f}`")

                                    # Branch path
                                    st.markdown(f"**🌳 Path:** {result.get('branch_path', 'N/A')}")
                        else:
                            st.success("✅ Tutte le triplette sopra la soglia!")

                    # Download risultati validati
                    json_output = json.dumps(results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="⬇️ Download Risultati Validazione (JSON)",
                        data=json_output,
                        file_name="validation_results.json",
                        mime="application/json"
                    )

                    # Segnala che le triplette validate sono pronte per il KG
                    st.info("💡 Triplette validate! Vai alla tab **Knowledge Graph Builder** per costruire il grafo")

    # Tab 3: Knowledge Graph Builder
    with tab3:
        st.header("🕸️ Knowledge Graph Builder")
        st.write("Costruisci il Knowledge Graph classificando le triplette in broad/narrow topics")

        # Inizializza il KG builder (cached)
        @st.cache_resource
        def init_kg_builder(_llm):
            """Inizializza il Knowledge Graph Builder."""
            from src.agents import KnowledgeGraphBuilder, InMemoryKnowledgeGraph
            storage = InMemoryKnowledgeGraph()
            return KnowledgeGraphBuilder(_llm, storage=storage)

        kg_builder = init_kg_builder(llm)

        # Mostra statistiche KG corrente
        st.subheader("📊 Statistiche Knowledge Graph")
        kg_stats = kg_builder.get_storage().get_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Broad Topics", kg_stats["num_broader_topics"])
        with col2:
            st.metric("Narrow Topics", kg_stats["num_narrower_topics"])
        with col3:
            st.metric("Triplette nel KG", kg_stats["num_triplets"])

        # Visualizza struttura KG
        if kg_stats["num_broader_topics"] > 0:
            with st.expander("🌳 Visualizza Struttura Knowledge Graph"):
                all_topics = kg_builder.get_storage().get_all_topics()
                for broader, narrowers in all_topics.items():
                    st.markdown(f"**{broader}**")
                    for narrower in narrowers:
                        st.markdown(f"  └─ {narrower}")

        st.markdown("---")

        # Carica triplette da processare
        st.subheader("📥 Carica Triplette")

        load_mode = st.radio(
            "Sorgente triplette:",
            ["Sessione corrente (dopo estrazione)", "Sessione corrente (dopo validazione)", "Carica file JSON custom"],
            horizontal=False
        )

        triplets_to_process = None
        source_info = ""

        # Carica da sessione corrente (estrazione)
        if load_mode == "Sessione corrente (dopo estrazione)":
            if 'extracted_triplets' in st.session_state and st.session_state['extracted_triplets']:
                triplets_to_process = st.session_state['extracted_triplets']
                source_info = "triplette estratte (non validate)"
            else:
                st.warning("⚠️ Nessuna tripletta estratta in memoria. Vai alla tab **Estrazione Triplette**")

        # Carica da sessione corrente (validazione)
        elif load_mode == "Sessione corrente (dopo validazione)":
            if 'validation_results' in st.session_state and st.session_state['validation_results']:
                # Le validation_results contengono le triplette con metadata ontologico
                triplets_to_process = st.session_state['validation_results']
                source_info = "triplette validate con ontologia"
            else:
                st.warning("⚠️ Nessuna tripletta validata in memoria. Vai alla tab **Ontology Validation**")

        # Carica da file JSON
        elif load_mode == "Carica file JSON custom":
            uploaded_file = st.file_uploader("Carica file JSON con triplette", type=['json'], key="kg_json_upload")

            if uploaded_file:
                try:
                    loaded_data = json.load(uploaded_file)

                    # Supporta diversi formati
                    if 'triplets' in loaded_data:
                        triplets_to_process = loaded_data['triplets']
                    elif isinstance(loaded_data, list):
                        triplets_to_process = loaded_data
                    else:
                        st.error("❌ Formato JSON non riconosciuto")

                    if triplets_to_process:
                        source_info = f"file {uploaded_file.name}"
                        st.success(f"✅ Caricato file con {len(triplets_to_process)} triplette")

                        with st.expander("🔍 Preview Triplette"):
                            st.json(triplets_to_process[:3])

                except json.JSONDecodeError as e:
                    st.error(f"❌ Errore nel parsing JSON: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Errore: {str(e)}")

        # Configurazione
        if triplets_to_process:
            st.success(f"✅ Trovate **{len(triplets_to_process)}** triplette da processare ({source_info})")

            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🛠️ KG Builder Config")

                ontology_check_enabled = st.checkbox(
                    "Abilita check ontologico",
                    value=True,
                    help="Se disabilitato, le triplette vengono processate senza validazione ontologica"
                )

            # Pulsante per costruire il KG
            if st.button("🚀 Costruisci Knowledge Graph", type="primary"):
                with st.spinner("⏳ Processamento triplette in corso..."):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    try:
                        # Esegui il builder
                        result = kg_builder.run(
                            triplets=triplets_to_process,
                            ontology_check_enabled=ontology_check_enabled
                        )

                        progress_bar.progress(1.0)

                        if result["success"]:
                            st.success("✅ Knowledge Graph costruito con successo!")

                            # Mostra statistiche aggiornate
                            updated_stats = result["kg_stats"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Broad Topics", updated_stats["num_broader_topics"])
                            with col2:
                                st.metric("Narrow Topics", updated_stats["num_narrower_topics"])
                            with col3:
                                st.metric("Triplette Totali", updated_stats["num_triplets"])

                            # Mostra triplette processate
                            processed = result["processed_triplets"]
                            st.subheader(f"📋 Triplette Processate: {len(processed)}")

                            for idx, triplet in enumerate(processed[:10], 1):
                                broader = triplet.get("broader_topic", "N/A")
                                narrower = triplet.get("narrower_topic", "N/A")

                                # Estrai valori (gestisce sia dict che stringa)
                                def get_value(field):
                                    val = triplet.get(field, "")
                                    return val.get("value", "") if isinstance(val, dict) else str(val)

                                subj = get_value("subject")
                                pred = get_value("predicate")
                                obj = get_value("object")

                                with st.expander(f"Tripletta #{idx}: {broader} → {narrower}"):
                                    st.markdown(f"**Topics:** `{broader}` → `{narrower}`")
                                    st.markdown(f"**Tripletta:** {subj} → {pred} → {obj}")

                                    # Metadata
                                    metadata = triplet.get("topic_metadata", {})
                                    if metadata:
                                        action = metadata.get("action", "N/A")
                                        st.markdown(f"**Action:** {action}")

                                    # Reasoning
                                    reasoning = triplet.get("classification_reasoning", "")
                                    if reasoning:
                                        st.markdown(f"**Reasoning:** {reasoning}")

                            if len(processed) > 10:
                                st.info(f"Mostrate prime 10 triplette. Totale: {len(processed)}")

                            # Mostra errori se presenti
                            errors = result.get("errors", [])
                            if errors:
                                with st.expander(f"⚠️ Errori ({len(errors)})"):
                                    for error in errors:
                                        st.warning(error)

                            # Download risultati
                            json_output = json.dumps(result, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="⬇️ Download Risultati KG Builder (JSON)",
                                data=json_output,
                                file_name="kg_builder_results.json",
                                mime="application/json"
                            )

                            # Salva in session state
                            st.session_state['kg_builder_results'] = result

                        else:
                            st.error(f"❌ Errore durante la costruzione del KG: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ Errore: {str(e)}")
                        import traceback
                        with st.expander("🐛 Stack Trace"):
                            st.code(traceback.format_exc())

    # Tab 4: Dati IoT
    with tab4:
        st.header("Analisi Dati IoT")
        st.write("Genera o inserisci dati IoT conformi all'ontologia")

        # Inizializza il generatore
        if "data_generator" not in st.session_state:
            st.session_state.data_generator = OntologyDataGenerator()

        generator = st.session_state.data_generator

        # Due modalità: Genera o Inserisci Manualmente
        mode = st.radio(
            "Modalità:",
            ["Genera da Ontologia", "Inserisci Manualmente"],
            horizontal=True
        )

        if mode == "Genera da Ontologia":
            st.subheader("Generazione Automatica Dati")

            col1, col2 = st.columns(2)

            with col1:
                device_type = st.selectbox(
                    "Tipo di dispositivo:",
                    generator.get_available_devices()
                )

                device_id = st.text_input(
                    "ID Dispositivo (opzionale):",
                    placeholder="Lascia vuoto per generazione automatica"
                )

            with col2:
                num_records = st.slider(
                    "Numero di record da generare:",
                    min_value=1,
                    max_value=50,
                    value=5
                )

                time_interval = st.slider(
                    "Intervallo tra record (minuti):",
                    min_value=1,
                    max_value=1440,
                    value=60
                )

            # Mostra info sul dispositivo
            with st.expander("Info Dispositivo e Metriche"):
                device_info = generator.get_device_metrics(device_type)
                st.write(f"**Sensori:** {', '.join(device_info['sensors'])}")
                st.write(f"**Metriche disponibili:** {len(device_info['metrics'])}")

                metrics_list = []
                for metric, range_val in device_info['metrics'].items():
                    if range_val:
                        metrics_list.append(f"- {metric}: {range_val}")
                    else:
                        metrics_list.append(f"- {metric}: (stringa)")

                st.text("\n".join(metrics_list[:10]))
                if len(metrics_list) > 10:
                    st.text(f"... e altre {len(metrics_list) - 10} metriche")

            if st.button("Genera e Invia Dati", key="generate_iot"):
                with st.spinner("Generazione dati..."):
                    try:
                        # Genera i dati
                        generated_data = generator.generate_data(
                            device_type=device_type,
                            device_id=device_id if device_id else None,
                            num_records=num_records,
                            time_interval_minutes=time_interval
                        )

                        st.success(f"Generati {len(generated_data)} record!")

                        # Mostra preview
                        with st.expander("Preview Dati Generati"):
                            st.json(generated_data[0])

                        # Invia al server MCP
                        mcp_config = config.get_mcp_config()
                        mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

                        success_count = 0
                        for record in generated_data:
                            try:
                                response = requests.post(
                                    f"{mcp_url}/api/iot/data",
                                    json=record,
                                    timeout=10
                                )
                                if response.status_code == 200:
                                    success_count += 1
                            except Exception as e:
                                st.warning(f"Errore invio record: {str(e)}")

                        if success_count == len(generated_data):
                            st.success(f"Tutti i {success_count} record inviati al server MCP!")
                        else:
                            st.warning(f"Inviati {success_count}/{len(generated_data)} record")

                    except Exception as e:
                        st.error(f"Errore: {str(e)}")

        else:
            # Modalità manuale
            st.subheader("Inserimento Manuale")

            device_type = st.selectbox(
                "Tipo di dispositivo:",
                ["fitbit", "garmin", "jawbone", "altro"]
            )

            device_id = st.text_input("ID Dispositivo:", placeholder="device_001")

            iot_data_input = st.text_area(
                "Dati IoT (formato JSON):",
                height=150,
                placeholder='{"heartrate": 75, "steps": 5000, ...}'
            )

            if st.button("Analizza Dati IoT", key="analyze_iot"):
                if device_id and iot_data_input:
                    try:
                        # Valida il JSON
                        iot_data = json.loads(iot_data_input)

                        with st.spinner("Invio dati al server MCP..."):
                            # Invia al server MCP
                            mcp_config = config.get_mcp_config()
                            mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

                            payload = {
                                "device_type": device_type,
                                "device_id": device_id,
                                "timestamp": datetime.now().isoformat(),
                                "data": iot_data
                            }

                            response = requests.post(
                                f"{mcp_url}/api/iot/data",
                                json=payload,
                                timeout=10
                            )

                            if response.status_code == 200:
                                st.success("Dati inviati al server MCP!")
                                st.json(response.json())

                                # Analizza con LLM
                                with st.spinner("Analisi in corso..."):
                                    messages = prompt_manager.build_messages(
                                        'iot_data_processing',
                                        iot_data=json.dumps(iot_data, indent=2)
                                    )

                                    llm_response = llm.generate_with_history(messages)

                                    st.subheader("Analisi:")
                                    st.write(llm_response)
                            else:
                                st.error(f"Errore server MCP: {response.status_code}")

                    except json.JSONDecodeError:
                        st.error("Formato JSON non valido")
                    except requests.exceptions.ConnectionError:
                        st.error("Impossibile connettersi al server MCP. Assicurati che sia in esecuzione.")
                    except Exception as e:
                        st.error(f"Errore: {str(e)}")
                else:
                    st.warning("Compila tutti i campi")

    # Tab 5: Servizi Esterni
    with tab5:
        st.header("Servizi Esterni")
        st.write("Integrazione con Gmail e altri servizi")

        service_type = st.selectbox(
            "Tipo di servizio:",
            ["gmail", "calendar", "altro"]
        )

        external_data_input = st.text_area(
            "Dati dal servizio (formato JSON):",
            height=150,
            placeholder='{"subject": "...", "body": "...", ...}'
        )

        if st.button("Invia Dati Esterni", key="send_external"):
            if external_data_input:
                try:
                    external_data = json.loads(external_data_input)

                    with st.spinner("Invio dati al server MCP..."):
                        mcp_config = config.get_mcp_config()
                        mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

                        payload = {
                            "source": service_type,
                            "data_id": "external_001",  # TODO: generare ID univoco
                            "timestamp": "2024-01-01T00:00:00Z",
                            "content": external_data
                        }

                        response = requests.post(
                            f"{mcp_url}/api/external/gmail",
                            json=payload,
                            timeout=10
                        )

                        if response.status_code == 200:
                            st.success("Dati inviati al server MCP!")
                            st.json(response.json())
                        else:
                            st.error(f"Errore server MCP: {response.status_code}")

                except json.JSONDecodeError:
                    st.error("Formato JSON non valido")
                except requests.exceptions.ConnectionError:
                    st.error("Impossibile connettersi al server MCP. Assicurati che sia in esecuzione.")
                except Exception as e:
                    st.error(f"Errore: {str(e)}")
            else:
                st.warning("Inserisci i dati da inviare")

    # Tab 6: Chat Agent con MCP
    with tab6:
        st.header("Chat Agent con Accesso MCP")
        st.write("Chatta con l'AI che può accedere autonomamente ai dati IoT tramite il server MCP")

        # Inizializza l'agent nella sessione
        if "mcp_agent" not in st.session_state:
            mcp_config = config.get_mcp_config()
            mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"
            st.session_state.mcp_agent = MCPAgent(llm, mcp_base_url=mcp_url)

        # Inizializza la cronologia della chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Mostra info sui tools disponibili
        with st.expander("Tools Disponibili per l'Agent"):
            agent = st.session_state.mcp_agent
            tools_info = agent.get_available_tools()
            for tool in tools_info:
                st.write(f"**{tool['name']}**: {tool['description']}")

        # Area per reset della conversazione
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Reset Conversazione"):
                st.session_state.mcp_agent.reset_conversation()
                st.session_state.chat_history = []
                st.success("Conversazione resettata!")
                st.rerun()

        # Visualizza la cronologia della chat
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)

        # Input dell'utente
        user_input = st.chat_input("Scrivi un messaggio all'agent...")

        if user_input:
            # Aggiungi messaggio utente alla cronologia
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Mostra il messaggio dell'utente
            with st.chat_message("user"):
                st.write(user_input)

            # Genera risposta dall'agent
            with st.chat_message("assistant"):
                with st.spinner("L'agent sta pensando..."):
                    try:
                        result = st.session_state.mcp_agent.chat(user_input)

                        response = result.get("response", "Errore nella generazione")
                        error = result.get("error")

                        if error:
                            st.error(f"Errore: {error}")
                        else:
                            st.write(response)

                            # Mostra quali tools sono stati usati (se disponibile)
                            tools_used = result.get("tools_used", [])
                            if tools_used:
                                with st.expander("Tools Utilizzati"):
                                    for tool in tools_used:
                                        st.write(f"- {tool}")

                        # Aggiungi risposta alla cronologia
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })

                    except Exception as e:
                        error_msg = f"Errore durante la generazione: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

        # Suggerimenti di esempio
        if not st.session_state.chat_history:
            st.markdown("### Esempi di domande:")
            st.markdown("""
            - "Quali dispositivi IoT sono disponibili?"
            - "Mostrami gli ultimi dati del mio smartwatch"
            - "Calcola le statistiche della mia frequenza cardiaca"
            - "Come sta la mia salute oggi?"
            """)


if __name__ == "__main__":
    main()
