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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Estrazione Triplette",
        "Dati IoT",
        "Servizi Esterni",
        "Chat Agent"
    ])

    # Tab 1: Estrazione Triplette da Testo
    with tab1:
        st.header("Estrazione Triplette da Testo")
        st.write("Inserisci del testo per estrarre triplette RDF (soggetto, predicato, oggetto)")

        text_input = st.text_area(
            "Testo da analizzare:",
            height=200,
            placeholder="Inserisci qui il testo..."
        )

        if st.button("Estrai Triplette", key="extract_triplets"):
            if text_input:
                with st.spinner("Estrazione in corso..."):
                    try:
                        # Costruisci i messaggi
                        messages = prompt_manager.build_messages(
                            'triplet_extraction',
                            text=text_input
                        )

                        # Genera la risposta
                        response = llm.generate_with_history(messages)

                        st.success("Estrazione completata!")
                        st.subheader("Risultato:")
                        st.code(response, language="json")

                    except Exception as e:
                        st.error(f"Errore durante l'estrazione: {str(e)}")
            else:
                st.warning("Inserisci del testo prima di procedere")

    # Tab 2: Dati IoT
    with tab2:
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

    # Tab 3: Servizi Esterni
    with tab3:
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

    # Tab 4: Chat Agent con MCP
    with tab4:
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
