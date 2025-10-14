# Design: Query Optimization for MCP Agent

## Problem
The current system passes large JSON objects to the agent, which causes:
- High token usage
- Slow response times
- Unnecessary data transfer
- Poor scalability with large datasets

## Root Causes
1. `get_user_context()` returns ALL aggregated data
2. `get_iot_recent_data()` returns full records
3. Agent doesn't know the data schema structure
4. No selective field querying capability

## Proposed Solution

### 1. Schema Discovery Tools
Add tools for the agent to discover available data structures:

- `get_data_schema()`: Returns the schema of available data fields
- `get_device_info(device_id)`: Returns metadata only (not data)

### 2. Selective Query Tools
Replace large data dumps with targeted queries:

- `query_iot_field(device_id, field_name, limit)`: Get specific field only
- `query_iot_range(device_id, start_time, end_time, fields)`: Time-based query with field selection
- `aggregate_iot_field(device_id, field_name, operation)`: Compute aggregations (avg, min, max, sum) without returning raw data

### 3. Summary Tools
Provide lightweight summaries instead of full data:

- `get_devices_summary()`: Device list with record counts only
- `get_latest_value(device_id, field_name)`: Single latest value
- `get_context_summary()`: High-level overview without raw data

### 4. Updated System Prompt
Teach the agent:
- Available data schemas
- How to discover data structure
- Query strategy (always start with schema/summary, then query specific fields)
- Best practices for minimal data retrieval

## Implementation Plan

### Phase 1: New MCP Server Endpoints
Add to `src/mcp/server.py`:
- `/api/schema/{device_id}` - Get data schema
- `/api/iot/field` - Query specific field
- `/api/iot/range` - Time-range query with field selection
- `/api/iot/aggregate` - Server-side aggregations
- `/api/iot/latest` - Single latest value

### Phase 2: New Langchain Tools
Add to `src/mcp/mcp_tools.py`:
- `get_data_schema(device_id)`
- `query_iot_field(device_id, field_name, limit)`
- `query_iot_range(device_id, start_time, end_time, fields)`
- `aggregate_iot_field(device_id, field_name, operation)`
- `get_latest_value(device_id, field_name)`

### Phase 3: Refactor Existing Tools
Modify in `src/mcp/mcp_tools.py`:
- `get_user_context()` - Return summary only, not full data
- `get_iot_recent_data()` - Add optional `fields` parameter for field selection
- `list_devices()` - Already lightweight, keep as is

### Phase 4: Enhanced System Prompt
Update `src/agents/prompts.py`:
- Add data schema documentation
- Add query strategy guidelines
- Add examples of efficient vs inefficient queries

## Benefits
1. **Reduced Token Usage**: 10-100x reduction by querying only needed fields
2. **Faster Responses**: Less data transfer and processing
3. **Better Scalability**: Works with millions of records
4. **Smarter Agent**: Learns to query efficiently
5. **Server-side Aggregation**: Compute statistics on server, not in LLM context

## Example Query Flow

**Before (inefficient):**
```
User: "What's my average heart rate today?"
Agent: calls get_user_context() → receives 10MB of JSON
Agent: parses all data in context to find heart rate values
```

**After (efficient):**
```
User: "What's my average heart rate today?"
Agent: calls get_devices_summary() → finds heart rate device
Agent: calls aggregate_iot_field(device_id="fitbit_123", field="heart_rate", operation="avg", time_filter="today")
Agent: receives single number: {"avg": 72.5}
```

## Migration Strategy
1. Add new endpoints and tools (non-breaking)
2. Update system prompt to prefer new tools
3. Monitor usage and token consumption
4. Deprecate old inefficient endpoints if needed
