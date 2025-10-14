from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uvicorn
from collections import defaultdict


class MCPServer:
    """Server MCP per esporre API che iniettano informazioni al modello."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Inizializza il server MCP.

        Args:
            host: Host su cui far girare il server
            port: Porta su cui far girare il server
        """
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="MCP Server - Human Digital Twin",
            description="API per l'interazione autonoma con il modello LLM",
            version="1.0.0"
        )

        # Storage in-memory per dati IoT e contesto
        self.iot_data_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.external_data_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Setup delle routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Configura le routes dell'API."""

        @self.app.get("/")
        async def root():
            """Endpoint di test."""
            return {"status": "ok", "message": "MCP Server is running"}

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        @self.app.post("/api/iot/data")
        async def receive_iot_data(data: IoTDataModel):
            """
            Riceve dati IoT dai dispositivi.

            Args:
                data: Dati IoT in formato JSON

            Returns:
                Conferma di ricezione con ID
            """
            # Salva i dati nello storage in-memory
            device_id = data.device_id
            data_dict = {
                "device_type": data.device_type,
                "device_id": device_id,
                "timestamp": data.timestamp,
                "data": data.data,
                "metadata": data.metadata
            }
            self.iot_data_store[device_id].append(data_dict)

            return {
                "status": "received",
                "device_id": device_id,
                "data_type": data.device_type,
                "timestamp": data.timestamp,
                "stored_count": len(self.iot_data_store[device_id])
            }

        @self.app.get("/api/iot/recent")
        async def get_recent_iot_data(
            device_id: str = Query(..., description="ID del dispositivo"),
            limit: int = Query(10, description="Numero massimo di record da restituire")
        ):
            """
            Recupera i dati IoT più recenti di un dispositivo.

            Args:
                device_id: ID del dispositivo
                limit: Numero massimo di record

            Returns:
                Lista di dati IoT recenti
            """
            if device_id not in self.iot_data_store:
                return {
                    "device_id": device_id,
                    "data": [],
                    "message": "Nessun dato disponibile per questo dispositivo"
                }

            # Recupera gli ultimi N record
            recent_data = self.iot_data_store[device_id][-limit:]

            return {
                "device_id": device_id,
                "count": len(recent_data),
                "data": recent_data
            }

        @self.app.get("/api/iot/stats")
        async def get_iot_stats(
            device_id: str = Query(..., description="ID del dispositivo")
        ):
            """
            Calcola statistiche aggregate sui dati IoT di un dispositivo.

            Args:
                device_id: ID del dispositivo

            Returns:
                Statistiche aggregate
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Dispositivo {device_id} non trovato")

            data_list = self.iot_data_store[device_id]

            if not data_list:
                return {"device_id": device_id, "stats": {}, "message": "Nessun dato"}

            # Calcola statistiche base
            stats = {
                "total_records": len(data_list),
                "first_timestamp": data_list[0]["timestamp"],
                "last_timestamp": data_list[-1]["timestamp"],
                "device_type": data_list[0]["device_type"]
            }

            # Calcola medie per campi numerici
            numeric_fields = {}
            for record in data_list:
                for key, value in record["data"].items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)

            averages = {}
            for field, values in numeric_fields.items():
                averages[field] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

            stats["metrics"] = averages

            return {
                "device_id": device_id,
                "stats": stats
            }

        @self.app.post("/api/external/gmail")
        async def receive_gmail_data(data: ExternalDataModel):
            """
            Riceve dati da Gmail o altri servizi esterni.

            Args:
                data: Dati esterni in formato JSON

            Returns:
                Conferma di ricezione
            """
            # Salva i dati esterni
            source = data.source
            data_dict = {
                "source": source,
                "data_id": data.data_id,
                "timestamp": data.timestamp,
                "content": data.content,
                "metadata": data.metadata
            }
            self.external_data_store[source].append(data_dict)

            return {
                "status": "received",
                "source": source,
                "data_id": data.data_id,
                "stored_count": len(self.external_data_store[source])
            }

        @self.app.get("/api/context")
        async def get_context():
            """
            Returns a lightweight summary of the current context.
            Does NOT return actual data, only metadata and counts.
            Use specific query tools to retrieve actual data.

            Returns:
                Context summary from various sources
            """
            # Aggregate metadata only (no actual data)
            context = {
                "iot_devices": list(self.iot_data_store.keys()),
                "external_sources": list(self.external_data_store.keys()),
                "total_iot_records": sum(len(v) for v in self.iot_data_store.values()),
                "total_external_records": sum(len(v) for v in self.external_data_store.values())
            }

            # Add lightweight device summary (metadata only, no data)
            iot_summary = {}
            for device_id, data_list in self.iot_data_store.items():
                if data_list:
                    latest = data_list[-1]
                    iot_summary[device_id] = {
                        "device_type": latest["device_type"],
                        "last_update": latest["timestamp"],
                        "available_fields": list(latest["data"].keys()),  # Field names only
                        "record_count": len(data_list)
                    }

            context["iot_summary"] = iot_summary

            return {
                "context": context,
                "sources": ["iot", "external"],
                "note": "This is a summary only. Use get_data_schema, query_iot_field, or aggregate_iot_field for actual data."
            }

        @self.app.get("/api/devices")
        async def list_devices():
            """
            Elenca tutti i dispositivi IoT registrati.

            Returns:
                Lista di device_id con informazioni base
            """
            devices = []
            for device_id, data_list in self.iot_data_store.items():
                if data_list:
                    devices.append({
                        "device_id": device_id,
                        "device_type": data_list[0]["device_type"],
                        "record_count": len(data_list),
                        "last_update": data_list[-1]["timestamp"]
                    })

            return {
                "count": len(devices),
                "devices": devices
            }

        @self.app.get("/api/schema/{device_id}")
        async def get_data_schema(device_id: str):
            """
            Returns the data schema for a specific device.
            Shows available fields and their types without returning actual data.

            Args:
                device_id: ID of the device

            Returns:
                Schema information with field names and types
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                return {"device_id": device_id, "schema": {}, "message": "No data available"}

            # Extract schema from the latest record
            latest_record = data_list[-1]
            schema = {
                "device_type": latest_record["device_type"],
                "fields": {}
            }

            # Analyze fields from multiple records to get accurate types
            for record in data_list[-10:]:  # Check last 10 records
                for field_name, value in record["data"].items():
                    field_type = type(value).__name__
                    if field_name not in schema["fields"]:
                        schema["fields"][field_name] = {
                            "type": field_type,
                            "sample_value": value
                        }

            return {
                "device_id": device_id,
                "schema": schema,
                "total_records": len(data_list)
            }

        @self.app.get("/api/iot/field")
        async def query_iot_field(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name to query"),
            limit: int = Query(10, description="Maximum number of records")
        ):
            """
            Query a specific field from IoT data, returning only that field.

            Args:
                device_id: ID of the device
                field_name: Name of the field to retrieve
                limit: Maximum number of records

            Returns:
                List of values for the specified field
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                return {"device_id": device_id, "field": field_name, "values": [], "message": "No data"}

            # Extract only the requested field
            values = []
            for record in data_list[-limit:]:
                if field_name in record["data"]:
                    values.append({
                        "timestamp": record["timestamp"],
                        "value": record["data"][field_name]
                    })

            return {
                "device_id": device_id,
                "field": field_name,
                "count": len(values),
                "values": values
            }

        @self.app.get("/api/iot/latest")
        async def get_latest_value(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name")
        ):
            """
            Get the latest value for a specific field.

            Args:
                device_id: ID of the device
                field_name: Name of the field

            Returns:
                Latest value with timestamp
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                raise HTTPException(404, f"No data available for device {device_id}")

            # Get latest record
            latest_record = data_list[-1]
            if field_name not in latest_record["data"]:
                raise HTTPException(404, f"Field {field_name} not found in device data")

            return {
                "device_id": device_id,
                "field": field_name,
                "value": latest_record["data"][field_name],
                "timestamp": latest_record["timestamp"]
            }

        @self.app.get("/api/iot/aggregate")
        async def aggregate_iot_field(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name to aggregate"),
            operation: str = Query(..., description="Operation: avg, min, max, sum, count")
        ):
            """
            Compute server-side aggregation on a field without returning raw data.

            Args:
                device_id: ID of the device
                field_name: Name of the field to aggregate
                operation: Aggregation operation (avg, min, max, sum, count)

            Returns:
                Aggregated value
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                raise HTTPException(404, f"No data available for device {device_id}")

            # Extract field values
            values = []
            for record in data_list:
                if field_name in record["data"]:
                    value = record["data"][field_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            if not values:
                raise HTTPException(404, f"No numeric values found for field {field_name}")

            # Compute aggregation
            result = None
            if operation == "avg":
                result = sum(values) / len(values)
            elif operation == "min":
                result = min(values)
            elif operation == "max":
                result = max(values)
            elif operation == "sum":
                result = sum(values)
            elif operation == "count":
                result = len(values)
            else:
                raise HTTPException(400, f"Invalid operation: {operation}. Use: avg, min, max, sum, count")

            return {
                "device_id": device_id,
                "field": field_name,
                "operation": operation,
                "result": result,
                "sample_size": len(values)
            }

    def run(self, debug: bool = False) -> None:
        """
        Avvia il server MCP.

        Args:
            debug: Se True, abilita il modo debug
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if debug else "info"
        )

    def get_app(self) -> FastAPI:
        """
        Restituisce l'app FastAPI per testing o deployment.

        Returns:
            L'istanza FastAPI
        """
        return self.app


# Modelli Pydantic per la validazione dei dati

class IoTDataModel(BaseModel):
    """Modello per i dati IoT."""
    device_type: str
    device_id: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ExternalDataModel(BaseModel):
    """Modello per i dati da servizi esterni."""
    source: str  # gmail, calendar, etc.
    data_id: str
    timestamp: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
