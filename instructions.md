Triple: (subject, predicate, object) + tipi Schema.org
Output: {broader_topic, narrower_topic} seguendo SKOS
```

**Output:** broader e narrower topic

---

## BLOCCO 2: Topic Merge Check
```
IF broader_topic esiste in indice:
    usa quello esistente
    IF narrower_topic esiste:
        vai a BLOCCO 3
    ELSE:
        crea narrower_topic
        salta a BLOCCO 6
ELSE:
    crea broader_topic e narrower_topic
    salta a BLOCCO 6
```

---

## BLOCCO 3: Entity Matching (LLM 2)

**Per ogni entità (subject, object) di tipo rilevante:**

**Prompt:**
```
New entity: X
Existing entities in narrower_topic: [Y1, Y2, ...]
Decision: SAME | RELATED | DIFFERENT
If RELATED: specify relation type
```

**Output:** decisione + eventuale relazione

---

## BLOCCO 4: Graph Update
```
FOR each entity:
    IF SAME:
        merge con entity esistente
    IF RELATED:
        crea nuovo nodo + arco relazione
    IF DIFFERENT:
        crea nuovo nodo
        
Aggiungi archi della tripletta
```

---

## BLOCCO 5: Index Update
```
topics[broader_topic][narrower_topic].append(triple_id)



ESEMPIO:
Nodi: [Marco, Running, 10km, Park]
Archi: 
- (Marco, runs, 10km)
- (Marco, runsAt, Park)
```

**Indice Topic:**
```
Physical Fitness:
  └─ Running: [triple_1, triple_2]
```

---

## NUOVA TRIPLETTA
```
(Marco, jogs, 5km)
```

---

## PROCESSO

### BLOCCO 1: Topic Classification (LLM 1)
**Output:** `{broader: "Physical Fitness", narrower: "Running"}`

### BLOCCO 2: Topic Check
`Physical Fitness/Running` **esiste** → vai a matching

### BLOCCO 3: Entity Matching (LLM 2)

**Entità da verificare:** `5km`

**Prompt:**
```
New: "5km" (QuantitativeValue)
Existing in Running: ["10km", "Park"]
Decision?
```

**Output:** `{decision: "DIFFERENT"}`

---

**Entità da verificare:** predicato `jogs`

**Prompt:**
```
New predicate: "jogs"
Existing predicates in Running: ["runs", "runsAt"]
Are "jogs" and "runs" the same action?
```

**Output:** `{decision: "SAME", merge_into: "runs"}`

---

## BLOCCO 4: Graph Update
```
- Crea nodo: 5km
- Merge predicato: jogs → runs
- Aggiungi arco: (Marco, runs, 5km)
```

---

## STATO FINALE

**Grafo:**
```
Nodi: [Marco, Running, 10km, 5km, Park]
Archi:
- (Marco, runs, 10km)
- (Marco, runs, 5km)  ← NUOVO
- (Marco, runsAt, Park)
```

**Indice Topic:**
```
Physical Fitness:
  └─ Running: [triple_1, triple_2, triple_3]  ← aggiunto