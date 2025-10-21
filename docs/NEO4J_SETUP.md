# Neo4j Setup Guide

## Installazione Neo4j Desktop

### 1. Download Neo4j Desktop

Scarica Neo4j Desktop da: https://neo4j.com/download/

- Scegli "Neo4j Desktop" per sviluppo locale
- Disponibile per Windows, macOS, Linux

### 2. Installazione

1. Esegui l'installer scaricato
2. Segui la procedura guidata
3. Avvia Neo4j Desktop

### 3. Creazione Database

1. Apri Neo4j Desktop
2. Clicca su **"New"** → **"Create project"**
3. Nel progetto, clicca **"Add"** → **"Local DBMS"**
4. Configura:
   - **Name**: `HumanDigitalTwin`
   - **Password**: Scegli una password (es. `humandigitaltwin123`)
   - **Version**: Usa l'ultima versione stabile (5.x)
5. Clicca **"Create"**

### 4. Avvio Database

1. Nel progetto, clicca sul database `HumanDigitalTwin`
2. Clicca **"Start"**
3. Aspetta che lo status diventi "Active" (verde)

## Configurazione Progetto

### 1. Aggiorna il file `.env`

Copia `.env.example` in `.env` (se non esiste già) e configura:

```bash
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=humandigitaltwin123  # Usa la password che hai impostato
NEO4J_DATABASE=neo4j
```

### 2. Configura `config.yaml`

Il file `config.yaml` è già configurato:

```yaml
knowledge_graph:
  storage_type: "neo4j"  # "neo4j" or "in_memory"
  neo4j:
    uri: "bolt://localhost:7687"
    database: "neo4j"
```

Per usare storage **in-memory** (temporaneo), cambia:

```yaml
knowledge_graph:
  storage_type: "in_memory"
```

### 3. Installa Dipendenze

```bash
pip install -r requirements.txt
```

Questo installerà:
- `neo4j==5.28.0` - Driver ufficiale Neo4j
- `langchain-neo4j==0.3.6` - Integrazione LangChain
- `networkx==3.5` - Per visualizzazioni
- `graphviz==0.21` - Per rendering grafici

## Verifica Connessione

### Opzione 1: Da Streamlit

1. Avvia Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Vai alla **Tab 3: Knowledge Graph Builder**

3. Dovresti vedere:
   - ✅ `Connesso a Neo4j: bolt://localhost:7687 (database: neo4j)`

4. Se vedi un errore:
   - ⚠️ Verifica che Neo4j sia **avviato** (status "Active" in Neo4j Desktop)
   - ⚠️ Controlla username/password in `.env`
   - ⚠️ Verifica che la porta `7687` sia libera

### Opzione 2: Da Python (Test rapido)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "humandigitaltwin123")
)

with driver.session() as session:
    result = session.run("RETURN 'Hello Neo4j!' AS message")
    print(result.single()["message"])

driver.close()
```

Se stampa `Hello Neo4j!`, la connessione funziona!

## Struttura Knowledge Graph

Il grafo Neo4j sarà strutturato così:

```
(:KnowledgeGraph {id: 'main'})
  ├─[:HAS_CATEGORY]→ (:BroaderTopic {name: "Health"})
  │   └─[:HAS_SUBCATEGORY]→ (:NarrowerTopic {name: "Vital Signs"})
  │       └─[:CONTAINS]→ (:Triplet {subject: "User", predicate: "heart_rate", object: "72 bpm"})
  │
  ├─[:HAS_CATEGORY]→ (:BroaderTopic {name: "Social"})
  │   └─[:HAS_SUBCATEGORY]→ (:NarrowerTopic {name: "Friendships"})
  │       └─[:CONTAINS]→ (:Triplet {subject: "John", predicate: "has_friend", object: "Mary"})
  └─ ...
```

### Query Esempio

Apri **Neo4j Browser** (da Neo4j Desktop, clicca "Open" accanto al database) e prova:

```cypher
// Visualizza tutto il grafo
MATCH (n) RETURN n LIMIT 100

// Conta nodi per tipo
MATCH (n) RETURN labels(n) AS type, count(n) AS count

// Trova triplette per topic
MATCH (b:BroaderTopic {name: "Health"})-[:HAS_SUBCATEGORY]->(n:NarrowerTopic)-[:CONTAINS]->(t:Triplet)
RETURN b.name, n.name, t.subject, t.predicate, t.object
LIMIT 10
```

## Troubleshooting

### Errore: "Connection refused"

- ✅ Verifica che Neo4j sia **avviato** (status "Active")
- ✅ Controlla la porta in `config.yaml` (`7687` di default)

### Errore: "Authentication failed"

- ✅ Controlla username/password in `.env`
- ✅ Username di default è sempre `neo4j`
- ✅ La password è quella scelta durante la creazione del database

### Errore: "Database not found"

- ✅ Il database di default si chiama `neo4j`
- ✅ Verifica `NEO4J_DATABASE` in `.env`

### Performance lente

- ✅ Crea indexes su Neo4j Browser:
  ```cypher
  CREATE INDEX broader_name IF NOT EXISTS FOR (b:BroaderTopic) ON (b.name)
  CREATE INDEX narrower_name IF NOT EXISTS FOR (n:NarrowerTopic) ON (n.name)
  ```

## Reset Database (se necessario)

Per cancellare tutti i dati:

```cypher
// Da Neo4j Browser
MATCH (n) DETACH DELETE n
```

⚠️ **ATTENZIONE**: Questo elimina TUTTO il grafo!

## Backup & Restore

### Backup

Da Neo4j Desktop:
1. Ferma il database
2. Clicca **"..."** → **"Dump"**
3. Salva il file `.dump`

### Restore

1. Ferma il database
2. Clicca **"..."** → **"Load from dump"**
3. Seleziona il file `.dump`
4. Riavvia il database
