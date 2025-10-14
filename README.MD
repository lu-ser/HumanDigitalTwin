# Human Digital Twin

Framework modulare per la creazione di un Digital Twin umano utilizzando LLM, server MCP e frontend Streamlit.

## Architettura

Il progetto è strutturato in moduli separati:

- **src/config**: Gestione configurazione da YAML e variabili d'ambiente
- **src/llm**: Wrapper generico per servizi LLM con implementazione Groq + function calling
- **src/prompts**: Gestione centralizzata dei prompt
- **src/mcp**: Server MCP con API REST per ingestione/recupero dati
- **src/agents**: Agent LLM con accesso autonomo al server MCP
- **src/data_generator**: Generatore dati IoT basato su ontologia
- **app.py**: Frontend Streamlit

## Setup

### 1. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 2. Configura l'ambiente

Copia il file `.env_example` in `.env` e inserisci la tua API key:

```bash
cp .env_example .env
```

Modifica `.env`:
```
GROQ_API_KEY=your_actual_api_key_here
```

### 3. Configura il modello

Modifica `config.yaml` per selezionare il modello LLM e altri parametri:

```yaml
llm:
  provider: "groq"
  model: "mixtral-8x7b-32768"
  temperature: 0.7
  max_tokens: 2000
```

## Utilizzo

### Avvia il server MCP

In un terminale:

```bash
python src/mcp/run_server.py
```

Il server sarà disponibile su `http://localhost:8000`

API docs automatiche: `http://localhost:8000/docs`

### Avvia il frontend Streamlit

In un altro terminale:

```bash
streamlit run app.py
```

Il frontend sarà disponibile su `http://localhost:8501`

## Funzionalità

### 1. Estrazione Triplette
Inserisci testo e genera triplette RDF (soggetto, predicato, oggetto)

### 2. Generazione e Analisi Dati IoT

**Generazione automatica**: Il sistema genera dati IoT realistici basati sull'ontologia `onto.owl`:
- **Fitbit**: Steps, sleep, heart rate, calories, activity levels
- **Garmin**: Heart rate, GPS, training metrics, activity performance
- **Jawbone**: Steps, active time, distance, activity percentage

I dati vengono inviati al server MCP e possono essere analizzati dall'LLM.

**Inserimento manuale**: Possibilità di inserire dati custom in formato JSON.

### 3. Servizi Esterni
Integrazione con Gmail e altri servizi attraverso il server MCP

### 4. Chat Agent con Function Calling

L'agent LLM può chiamare autonomamente il server MCP per recuperare:
- Dati recenti da dispositivi IoT
- Statistiche aggregate (medie, min, max)
- Contesto completo dell'utente
- Lista dispositivi disponibili

L'agent decide autonomamente quando recuperare informazioni dal server MCP.

## Architettura MCP con Function Calling

Il sistema implementa un vero **Model Context Protocol** dove l'LLM può chiamare autonomamente le API del server MCP:

1. **Utente fa una domanda** (es: "Come sta la mia salute?")
2. **Agent LLM analizza** e decide di aver bisogno di dati IoT
3. **Agent chiama autonomamente** il tool `get_iot_statistics`
4. **Server MCP risponde** con i dati aggregati
5. **Agent analizza i dati** e genera una risposta per l'utente

### API MCP Disponibili

**Endpoint GET** (usati dall'agent):
- `/api/iot/recent?device_id=X&limit=10` - Dati recenti
- `/api/iot/stats?device_id=X` - Statistiche aggregate
- `/api/context` - Contesto completo utente
- `/api/devices` - Lista dispositivi

**Endpoint POST** (usati dal frontend):
- `/api/iot/data` - Invia nuovi dati IoT
- `/api/external/gmail` - Invia dati da servizi esterni

## Estendibilità

### Aggiungere un nuovo LLM provider

1. Crea una nuova classe in `src/llm/` che eredita da `BaseLLM`
2. Implementa i metodi astratti
3. Registra il provider nel `LLMFactory`

```python
from src.llm import LLMFactory, BaseLLM

class MyCustomLLM(BaseLLM):
    # ... implementazione

LLMFactory.register_provider('my_provider', MyCustomLLM)
```

### Aggiungere nuovi prompt

Modifica `src/prompts/prompts.yaml`:

```yaml
my_new_prompt:
  system: |
    System prompt here
  user_template: |
    User template with {placeholders}
```

### Aggiungere nuovi endpoint MCP

Modifica `src/mcp/server.py` aggiungendo nuove route nell'`_setup_routes()`.

### Aggiungere nuovi dispositivi IoT

Modifica `src/data_generator/ontology_generator.py` aggiungendo un nuovo profilo in `device_profiles`.

## Struttura del Progetto

```
HumanDigitalTwin/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_manager.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base_llm.py
│   │   ├── groq_llm.py (con function calling)
│   │   └── llm_factory.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── prompt_manager.py
│   │   └── prompts.yaml
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py (GET + POST endpoints)
│   │   ├── mcp_tools.py (Langchain tools)
│   │   └── run_server.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── mcp_agent.py (Agent con function calling)
│   └── data_generator/
│       ├── __init__.py
│       └── ontology_generator.py (Genera dati da onto.owl)
├── ontologies/
│   └── onto.owl (Ontologia dispositivi IoT)
├── app.py
├── config.yaml
├── requirements.txt
├── .env
├── .env_example
└── README.MD
```

## Esempio di Utilizzo Completo

```bash
# 1. Avvia il server MCP
python src/mcp/run_server.py

# 2. Avvia Streamlit
streamlit run app.py

# 3. Nel browser:
#    - Tab "Dati IoT" → Genera dati Fitbit (5 record)
#    - Tab "Chat Agent" → Chiedi: "Mostrami le statistiche del mio Fitbit"
#    - L'agent chiamerà automaticamente l'MCP e risponderà con le statistiche!
```

## Flow Dati Completo

```
1. GENERAZIONE DATI
   Ontologia onto.owl → OntologyDataGenerator → Dati IoT realistici

2. STORAGE
   Dati IoT → POST /api/iot/data → Server MCP → Storage in-memory

3. INTERROGAZIONE AGENT
   Utente → Chat Agent → LLM decide tools da usare → GET /api/iot/* → MCP

4. RISPOSTA
   MCP → Dati → Agent LLM → Analisi → Risposta utente
```
