"""
Langchain Tools per interagire con il server MCP.
Questi tools permettono all'LLM di chiamare autonomamente l'MCP.
"""

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests
import json


# URL base del server MCP (può essere configurato)
_MCP_BASE_URL = "http://localhost:8000"


def set_mcp_base_url(url: str):
    """Imposta l'URL base del server MCP."""
    global _MCP_BASE_URL
    _MCP_BASE_URL = url.rstrip('/')


def _make_request(endpoint: str, method: str = "GET", **kwargs):
    """
    Effettua una richiesta HTTP al server MCP.

    Args:
        endpoint: Endpoint da chiamare (senza slash iniziale)
        method: Metodo HTTP
        **kwargs: Parametri aggiuntivi per requests

    Returns:
        Risposta JSON del server
    """
    url = f"{_MCP_BASE_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"Metodo HTTP non supportato: {method}")

        response.raise_for_status()
        result = response.json()

        # Log richiesta MCP
        try:
            from ..utils import get_logger
            logger = get_logger()
            logger.log_mcp_request(method, endpoint, kwargs.get('params'), result)
        except:
            pass  # Ignora errori di logging

        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status": "failed"}


# Definisci gli schemi Pydantic per i parametri dei tools

class IoTRecentDataInput(BaseModel):
    """Schema per get_iot_recent_data."""
    device_id: str = Field(description="ID del dispositivo IoT (es: 'fitbit_1234', 'garmin_5678')")
    limit: int = Field(default=10, description="Numero massimo di record da recuperare")


class IoTStatisticsInput(BaseModel):
    """Schema per get_iot_statistics."""
    device_id: str = Field(description="ID del dispositivo IoT")


@tool(args_schema=IoTRecentDataInput)
def get_iot_recent_data(device_id: str, limit: int = 10) -> str:
    """
    Recupera i dati IoT più recenti di un dispositivo specifico.
    Utile per vedere le ultime letture dei sensori.
    """
    result = _make_request(
        "api/iot/recent",
        params={"device_id": device_id, "limit": limit}
    )
    return json.dumps(result, indent=2)


@tool(args_schema=IoTStatisticsInput)
def get_iot_statistics(device_id: str) -> str:
    """
    Calcola statistiche aggregate sui dati IoT di un dispositivo.
    Include medie, min, max per tutti i parametri numerici (es: frequenza cardiaca, passi, temperatura).
    """
    result = _make_request("api/iot/stats", params={"device_id": device_id})
    return json.dumps(result, indent=2)


@tool
def get_user_context() -> str:
    """
    Get a lightweight summary of user context from all sources.
    Returns metadata and available fields, but NOT actual data values.
    Use this to discover what devices and fields are available, then use specific query tools for data.
    """
    result = _make_request("api/context")
    return json.dumps(result, indent=2)


@tool
def list_devices() -> str:
    """
    Elenca tutti i dispositivi IoT registrati nel sistema.
    Utile per scoprire quali dispositivi sono disponibili prima di interrogarli.
    """
    result = _make_request("api/devices")
    return json.dumps(result, indent=2)


class DataSchemaInput(BaseModel):
    """Schema per get_data_schema."""
    device_id: str = Field(description="ID del dispositivo IoT")


@tool(args_schema=DataSchemaInput)
def get_data_schema(device_id: str) -> str:
    """
    Returns the data schema for a specific device, showing available fields and their types.
    Use this FIRST to discover what data fields are available before querying specific data.
    This is lightweight and doesn't return actual data, only the structure.
    """
    result = _make_request(f"api/schema/{device_id}")
    return json.dumps(result, indent=2)


class QueryFieldInput(BaseModel):
    """Schema per query_iot_field."""
    device_id: str = Field(description="ID del dispositivo IoT")
    field_name: str = Field(description="Nome del campo da recuperare (es: 'heart_rate', 'steps', 'temperature')")
    limit: int = Field(default=10, description="Numero massimo di valori da recuperare")


@tool(args_schema=QueryFieldInput)
def query_iot_field(device_id: str, field_name: str, limit: int = 10) -> str:
    """
    Query a SPECIFIC field from IoT data, returning only that field's values.
    Much more efficient than getting all data when you only need one field.
    Example: To get only heart rate data, use field_name='heart_rate'
    """
    result = _make_request(
        "api/iot/field",
        params={"device_id": device_id, "field_name": field_name, "limit": limit}
    )
    return json.dumps(result, indent=2)


class LatestValueInput(BaseModel):
    """Schema per get_latest_value."""
    device_id: str = Field(description="ID del dispositivo IoT")
    field_name: str = Field(description="Nome del campo (es: 'heart_rate', 'steps')")


@tool(args_schema=LatestValueInput)
def get_latest_value(device_id: str, field_name: str) -> str:
    """
    Get the most recent value for a specific field.
    Most efficient way to check current status of a single metric.
    Example: Get current heart rate with device_id='fitbit_123', field_name='heart_rate'
    """
    result = _make_request(
        "api/iot/latest",
        params={"device_id": device_id, "field_name": field_name}
    )
    return json.dumps(result, indent=2)


class AggregateFieldInput(BaseModel):
    """Schema per aggregate_iot_field."""
    device_id: str = Field(description="ID del dispositivo IoT")
    field_name: str = Field(description="Nome del campo da aggregare")
    operation: str = Field(description="Operazione: avg, min, max, sum, count")


@tool(args_schema=AggregateFieldInput)
def aggregate_iot_field(device_id: str, field_name: str, operation: str) -> str:
    """
    Compute aggregations (avg, min, max, sum, count) on a field WITHOUT returning raw data.
    Most efficient for statistical queries. The computation happens on the server.
    Example: Get average heart rate with operation='avg', field_name='heart_rate'
    Available operations: avg, min, max, sum, count
    """
    result = _make_request(
        "api/iot/aggregate",
        params={"device_id": device_id, "field_name": field_name, "operation": operation}
    )
    return json.dumps(result, indent=2)


def get_mcp_tools(mcp_base_url: str = "http://localhost:8000") -> list:
    """
    Restituisce la lista di tutti i tools MCP per Langchain.

    Args:
        mcp_base_url: URL base del server MCP

    Returns:
        Lista di tools Langchain
    """
    # Imposta l'URL globale
    set_mcp_base_url(mcp_base_url)

    # Restituisci i tools (nuovi tools ottimizzati + tools legacy)
    return [
        # Prioritized efficient tools
        list_devices,           # Start here to discover devices
        get_data_schema,        # Then discover available fields
        get_latest_value,       # Most efficient for single values
        aggregate_iot_field,    # Most efficient for statistics
        query_iot_field,        # Efficient for specific fields

        # Legacy tools (less efficient, but still available)
        get_iot_recent_data,    # Returns full records
        get_iot_statistics,     # Returns all stats
        get_user_context,       # Returns all context
    ]
