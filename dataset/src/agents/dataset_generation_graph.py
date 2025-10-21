"""
LangGraph pipeline for synthetic life scene dataset generation.
Generates realistic life scenes with device data, calendar events, messages, and health deficiency analysis.
"""

from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import operator
import json
import yaml
from pathlib import Path


# ===== Pydantic Models for Structured Output =====

class Scene(BaseModel):
    """A single life scene."""
    scene_id: int = Field(description="Scene number")
    day: str = Field(description="Day of the week")
    time: str = Field(description="Time of day (morning/afternoon/evening/night)")
    description: str = Field(description="Detailed scene description")


class DeviceData(BaseModel):
    """Device sensor data for a scene."""
    smartwatch: Dict[str, Any] = Field(default_factory=dict, description="Smartwatch data")
    gps: Dict[str, Any] = Field(default_factory=dict, description="GPS data")
    smartphone: Dict[str, Any] = Field(default_factory=dict, description="Smartphone data")
    smartring: Dict[str, Any] = Field(default_factory=dict, description="Smartring data")
    smart_home: Dict[str, Any] = Field(default_factory=dict, description="Smart home sensors data")


class CalendarEvent(BaseModel):
    """A single calendar event."""
    title: str
    start_time: str
    end_time: str
    location: str
    description: str = ""


class CalendarEvents(BaseModel):
    """List of calendar events."""
    events: List[CalendarEvent] = Field(default_factory=list)


class Message(BaseModel):
    """A single message."""
    platform: str
    direction: str
    contact: str
    timestamp: str
    content: str


class Messages(BaseModel):
    """List of messages."""
    messages: List[Message] = Field(default_factory=list)


class DeficiencyStatus(BaseModel):
    """Status for a health category."""
    status: str = Field(description="OK/WARNING/INSUFFICIENT/UNKNOWN/HIGH")
    reason: str = Field(description="Explanation")


class Deficiencies(BaseModel):
    """Health deficiency analysis."""
    physical_activity: DeficiencyStatus
    sleep: DeficiencyStatus
    stress: DeficiencyStatus
    workload: DeficiencyStatus
    social_interaction: DeficiencyStatus
    environment: DeficiencyStatus
    emotional_tone: DeficiencyStatus
    digital_behavior: DeficiencyStatus


# ===== Graph State =====

class GraphState(TypedDict):
    """
    State for the dataset generation graph.

    Fields:
    - profile: Input profile text
    - num_scenes: Number of scenes to generate
    - current_scene_index: Current scene being processed
    - scenes: List of generated scenes
    - device_data: List of device data for each scene
    - calendar_events: List of calendar events for each scene
    - messages: List of messages for each scene
    - deficiencies: List of deficiency analyses for each scene
    - final_dataset: Compiled final dataset with all data merged into scenes
    - error: Error message if any
    """
    profile: str
    num_scenes: int
    current_scene_index: int
    scenes: Annotated[List[Dict[str, Any]], operator.add]
    device_data: Annotated[List[Dict[str, Any]], operator.add]
    calendar_events: Annotated[List[Dict[str, Any]], operator.add]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    deficiencies: Annotated[List[Dict[str, Any]], operator.add]
    final_dataset: Dict[str, Any]
    error: str


class DatasetGenerationGraph:
    """
    LangGraph pipeline for synthetic dataset generation.

    Pipeline:
    1. scene_generation (loop N times) - Generate life scenes from profile
    2. device_data_generation - Generate device sensor data
    3. calendar_generation - Generate calendar events
    4. messaging_generation - Generate messaging app data
    5. deficiency_detection - Detect health deficiencies using tools
    6. finalize - Compile final dataset
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.7,
        enable_logging: bool = True
    ):
        """
        Initialize the dataset generation graph.

        Args:
            llm_api_key: API key for Groq
            llm_model: LLM model to use
            temperature: Temperature for LLM
            enable_logging: Enable detailed logging
        """
        self.llm = ChatGroq(
            groq_api_key=llm_api_key,
            model_name=llm_model,
            temperature=temperature
        )
        self.enable_logging = enable_logging

        # Load configs
        config_dir = Path(__file__).parent.parent.parent / "config"

        with open(config_dir / "devices.yaml", 'r', encoding='utf-8') as f:
            self.devices_config = yaml.safe_load(f)

        with open(config_dir / "deficiencies.yaml", 'r', encoding='utf-8') as f:
            self.deficiencies_config = yaml.safe_load(f)

        # Prompt Manager
        from prompts import PromptManager
        self.prompt_manager = PromptManager()

        # Health Tools
        from tools.health_tools import get_health_tools
        self.health_tools = get_health_tools()

        # Build graph
        self.graph = self._build_graph()

        # Save diagram
        self._save_graph_diagram()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("scene_generation", self._scene_generation_node)
        workflow.add_node("device_data_generation", self._device_data_generation_node)
        workflow.add_node("calendar_generation", self._calendar_generation_node)
        workflow.add_node("messaging_generation", self._messaging_generation_node)
        workflow.add_node("deficiency_detection", self._deficiency_detection_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("scene_generation")

        # Scene generation loop
        workflow.add_conditional_edges(
            "scene_generation",
            self._should_continue_scenes,
            {
                "continue": "scene_generation",  # Loop back for next scene
                "next": "device_data_generation"  # All scenes generated
            }
        )

        # Linear flow after scene generation
        workflow.add_edge("device_data_generation", "calendar_generation")
        workflow.add_edge("calendar_generation", "messaging_generation")
        workflow.add_edge("messaging_generation", "deficiency_detection")
        workflow.add_edge("deficiency_detection", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _save_graph_diagram(self) -> None:
        """Save graph diagram as PNG."""
        try:
            assets_dir = Path(__file__).parent.parent.parent / "assets"
            assets_dir.mkdir(exist_ok=True)

            output_path = assets_dir / "dataset_graph.png"
            png_data = self.graph.get_graph().draw_mermaid_png()

            with open(output_path, "wb") as f:
                f.write(png_data)

            if self.enable_logging:
                print(f"[INFO] Graph diagram saved to: {output_path}")

        except Exception as e:
            if self.enable_logging:
                print(f"[WARNING] Could not save graph diagram: {str(e)}")

    def _scene_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node 1: Generate a single life scene from the profile.

        Args:
            state: Current graph state

        Returns:
            Updated state with new scene
        """
        profile = state["profile"]
        scene_number = state["current_scene_index"] + 1
        total_scenes = state["num_scenes"]
        previous_scenes = state.get("scenes", [])

        if self.enable_logging:
            print(f"\n[SCENE GENERATION] Generating scene {scene_number}/{total_scenes}")

        # Format previous scenes for context
        previous_scenes_text = ""
        if previous_scenes:
            previous_scenes_text = "\n\nPrevious scenes:\n"
            for scene in previous_scenes:
                previous_scenes_text += f"- Scene {scene['scene_id']} ({scene['day']} {scene['time']}): {scene['description'][:150]}...\n"

        # Build messages
        messages = self.prompt_manager.build_messages(
            'scene_generation',
            profile=profile,
            scene_number=scene_number,
            total_scenes=total_scenes,
            previous_scenes=previous_scenes_text
        )

        # Convert to Langchain format
        lc_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                lc_messages.append(SystemMessage(content=msg['content']))
            else:
                lc_messages.append(HumanMessage(content=msg['content']))

        try:
            # Use structured output
            llm_with_structure = self.llm.with_structured_output(Scene, method="json_mode")
            scene = llm_with_structure.invoke(lc_messages)

            scene_dict = {
                "scene_id": scene.scene_id,
                "day": scene.day,
                "time": scene.time,
                "description": scene.description
            }

            if self.enable_logging:
                print(f"[SCENE] {scene.day} {scene.time}: {scene.description[:100]}...")

            return {
                "scenes": [scene_dict],
                "current_scene_index": scene_number
            }

        except Exception as e:
            error_msg = f"Error generating scene {scene_number}: {str(e)}"
            if self.enable_logging:
                print(f"[ERROR] {error_msg}")
            return {
                "error": error_msg,
                "current_scene_index": scene_number
            }

    def _device_data_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node 2: Generate device sensor data for all scenes.

        Args:
            state: Current graph state

        Returns:
            Updated state with device data
        """
        scenes = state["scenes"]

        if self.enable_logging:
            print(f"\n[DEVICE DATA] Generating device data for {len(scenes)} scenes")

        all_device_data = []

        for scene in scenes:
            # Build messages
            devices_config_str = yaml.dump(self.devices_config, default_flow_style=False)
            messages = self.prompt_manager.build_messages(
                'device_data_generation',
                devices_config=devices_config_str,
                scene_description=f"{scene['day']} {scene['time']}: {scene['description']}"
            )

            # Convert to Langchain format
            lc_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                else:
                    lc_messages.append(HumanMessage(content=msg['content']))

            try:
                # Use structured output
                llm_with_structure = self.llm.with_structured_output(DeviceData, method="json_mode")
                device_data = llm_with_structure.invoke(lc_messages)

                device_dict = {
                    "scene_id": scene['scene_id'],
                    "smartwatch": device_data.smartwatch,
                    "gps": device_data.gps,
                    "smartphone": device_data.smartphone,
                    "smartring": device_data.smartring,
                    "smart_home": device_data.smart_home
                }

                all_device_data.append(device_dict)

                if self.enable_logging:
                    print(f"[DEVICE] Scene {scene['scene_id']}: Generated data for {len([k for k, v in device_dict.items() if k != 'scene_id' and v])} devices")

            except Exception as e:
                if self.enable_logging:
                    print(f"[ERROR] Device data for scene {scene['scene_id']}: {str(e)}")
                all_device_data.append({"scene_id": scene['scene_id'], "error": str(e)})

        return {"device_data": all_device_data}

    def _calendar_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node 3: Generate calendar events for all scenes.

        Args:
            state: Current graph state

        Returns:
            Updated state with calendar events
        """
        scenes = state["scenes"]

        if self.enable_logging:
            print(f"\n[CALENDAR] Generating calendar events for {len(scenes)} scenes")

        all_calendar_events = []

        for scene in scenes:
            # Build messages
            messages = self.prompt_manager.build_messages(
                'calendar_generation',
                scene_description=f"{scene['day']} {scene['time']}: {scene['description']}"
            )

            # Convert to Langchain format
            lc_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                else:
                    lc_messages.append(HumanMessage(content=msg['content']))

            try:
                # Use structured output
                llm_with_structure = self.llm.with_structured_output(CalendarEvents, method="json_mode")
                calendar_events = llm_with_structure.invoke(lc_messages)

                events_dict = {
                    "scene_id": scene['scene_id'],
                    "events": [
                        {
                            "title": e.title,
                            "start_time": e.start_time,
                            "end_time": e.end_time,
                            "location": e.location,
                            "description": e.description
                        }
                        for e in calendar_events.events
                    ]
                }

                all_calendar_events.append(events_dict)

                if self.enable_logging:
                    print(f"[CALENDAR] Scene {scene['scene_id']}: {len(calendar_events.events)} events")

            except Exception as e:
                if self.enable_logging:
                    print(f"[ERROR] Calendar for scene {scene['scene_id']}: {str(e)}")
                all_calendar_events.append({"scene_id": scene['scene_id'], "events": [], "error": str(e)})

        return {"calendar_events": all_calendar_events}

    def _messaging_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node 4: Generate messaging app data for all scenes.

        Args:
            state: Current graph state

        Returns:
            Updated state with messages
        """
        scenes = state["scenes"]
        profile = state["profile"]

        if self.enable_logging:
            print(f"\n[MESSAGING] Generating messages for {len(scenes)} scenes")

        all_messages = []

        for scene in scenes:
            # Build messages
            messages = self.prompt_manager.build_messages(
                'messaging_generation',
                scene_description=f"{scene['day']} {scene['time']}: {scene['description']}",
                profile=profile
            )

            # Convert to Langchain format
            lc_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                else:
                    lc_messages.append(HumanMessage(content=msg['content']))

            try:
                # Use structured output
                llm_with_structure = self.llm.with_structured_output(Messages, method="json_mode")
                msg_data = llm_with_structure.invoke(lc_messages)

                messages_dict = {
                    "scene_id": scene['scene_id'],
                    "messages": [
                        {
                            "platform": m.platform,
                            "direction": m.direction,
                            "contact": m.contact,
                            "timestamp": m.timestamp,
                            "content": m.content
                        }
                        for m in msg_data.messages
                    ]
                }

                all_messages.append(messages_dict)

                if self.enable_logging:
                    print(f"[MESSAGING] Scene {scene['scene_id']}: {len(msg_data.messages)} messages")

            except Exception as e:
                if self.enable_logging:
                    print(f"[ERROR] Messages for scene {scene['scene_id']}: {str(e)}")
                all_messages.append({"scene_id": scene['scene_id'], "messages": [], "error": str(e)})

        return {"messages": all_messages}

    def _deficiency_detection_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node 5: Detect health deficiencies using tools.

        Args:
            state: Current graph state

        Returns:
            Updated state with deficiency analysis
        """
        scenes = state["scenes"]
        device_data_list = state["device_data"]
        calendar_events_list = state["calendar_events"]
        messages_list = state["messages"]

        if self.enable_logging:
            print(f"\n[DEFICIENCY DETECTION] Analyzing {len(scenes)} scenes")

        all_deficiencies = []

        for i, scene in enumerate(scenes):
            scene_id = scene['scene_id']

            # Find corresponding data
            device_data = next((d for d in device_data_list if d.get('scene_id') == scene_id), {})
            calendar_events = next((c for c in calendar_events_list if c.get('scene_id') == scene_id), {})
            messages = next((m for m in messages_list if m.get('scene_id') == scene_id), {})

            # Build messages
            deficiencies_config_str = yaml.dump(self.deficiencies_config, default_flow_style=False)
            prompt_messages = self.prompt_manager.build_messages(
                'deficiency_detection',
                scene_description=f"{scene['day']} {scene['time']}: {scene['description']}",
                device_data=json.dumps(device_data, indent=2),
                calendar_events=json.dumps(calendar_events.get('events', []), indent=2),
                messages=json.dumps(messages.get('messages', []), indent=2),
                deficiencies_config=deficiencies_config_str
            )

            # Convert to Langchain format
            lc_messages = []
            for msg in prompt_messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                else:
                    lc_messages.append(HumanMessage(content=msg['content']))

            try:
                # Use LLM with tools
                llm_with_tools = self.llm.bind_tools(self.health_tools)

                # First call - may use tools
                response = llm_with_tools.invoke(lc_messages)

                # If tools were called, execute them and get final response
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    lc_messages.append(response)

                    # Execute each tool call
                    from langchain_core.messages import ToolMessage
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']

                        # Safety check: skip tools with null/None values or invalid types
                        has_null_values = any(v is None for v in tool_args.values())
                        if has_null_values:
                            if self.enable_logging:
                                print(f"[WARNING] Skipping tool {tool_name} with null values: {tool_args}")
                            lc_messages.append(
                                ToolMessage(
                                    content=f"ERROR: Cannot validate {tool_name} - data not available in this scene",
                                    tool_call_id=tool_call['id']
                                )
                            )
                            continue

                        # Convert string numbers to proper types
                        cleaned_args = {}
                        for key, value in tool_args.items():
                            if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                                # Convert string numbers to int or float
                                cleaned_args[key] = int(value) if '.' not in value else float(value)
                            else:
                                cleaned_args[key] = value

                        # Find and execute tool
                        selected_tool = next((t for t in self.health_tools if t.name == tool_name), None)
                        if selected_tool:
                            try:
                                tool_result = selected_tool.invoke(cleaned_args)
                                lc_messages.append(
                                    ToolMessage(
                                        content=str(tool_result),
                                        tool_call_id=tool_call['id']
                                    )
                                )
                            except Exception as tool_error:
                                if self.enable_logging:
                                    print(f"[WARNING] Tool {tool_name} error: {tool_error}")
                                lc_messages.append(
                                    ToolMessage(
                                        content=f"ERROR: {str(tool_error)}",
                                        tool_call_id=tool_call['id']
                                    )
                                )

                    # Get final response after tool execution
                    llm_with_structure = self.llm.with_structured_output(Deficiencies, method="json_mode")
                    final_response = llm_with_structure.invoke(lc_messages)
                else:
                    # No tools used, get structured output directly
                    llm_with_structure = self.llm.with_structured_output(Deficiencies, method="json_mode")
                    final_response = llm_with_structure.invoke(lc_messages)

                deficiencies_dict = {
                    "scene_id": scene_id,
                    "physical_activity": {
                        "status": final_response.physical_activity.status,
                        "reason": final_response.physical_activity.reason
                    },
                    "sleep": {
                        "status": final_response.sleep.status,
                        "reason": final_response.sleep.reason
                    },
                    "stress": {
                        "status": final_response.stress.status,
                        "reason": final_response.stress.reason
                    },
                    "workload": {
                        "status": final_response.workload.status,
                        "reason": final_response.workload.reason
                    },
                    "social_interaction": {
                        "status": final_response.social_interaction.status,
                        "reason": final_response.social_interaction.reason
                    },
                    "environment": {
                        "status": final_response.environment.status,
                        "reason": final_response.environment.reason
                    },
                    "emotional_tone": {
                        "status": final_response.emotional_tone.status,
                        "reason": final_response.emotional_tone.reason
                    },
                    "digital_behavior": {
                        "status": final_response.digital_behavior.status,
                        "reason": final_response.digital_behavior.reason
                    }
                }

                all_deficiencies.append(deficiencies_dict)

                if self.enable_logging:
                    issues = [k for k, v in deficiencies_dict.items() if k != 'scene_id' and v['status'] in ['WARNING', 'INSUFFICIENT', 'HIGH']]
                    print(f"[DEFICIENCY] Scene {scene_id}: {len(issues)} issues detected")

            except Exception as e:
                if self.enable_logging:
                    print(f"[ERROR] Deficiency detection for scene {scene_id}: {str(e)}")
                all_deficiencies.append({"scene_id": scene_id, "error": str(e)})

        return {"deficiencies": all_deficiencies}

    def _finalize_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node 6: Finalize and compile complete dataset.

        Args:
            state: Current graph state

        Returns:
            Final state
        """
        if self.enable_logging:
            print(f"\n[FINALIZE] Compiling final dataset")

        scenes = state["scenes"]
        device_data = state["device_data"]
        calendar_events = state["calendar_events"]
        messages = state["messages"]
        deficiencies = state["deficiencies"]

        # Compile full dataset
        dataset = {
            "profile": state["profile"],
            "num_scenes": len(scenes),
            "scenes": []
        }

        for scene in scenes:
            scene_id = scene['scene_id']

            scene_data = {
                **scene,
                "device_data": next((d for d in device_data if d.get('scene_id') == scene_id), {}),
                "calendar_events": next((c.get('events', []) for c in calendar_events if c.get('scene_id') == scene_id), []),
                "messages": next((m.get('messages', []) for m in messages if m.get('scene_id') == scene_id), []),
                "deficiencies": next((d for d in deficiencies if d.get('scene_id') == scene_id), {})
            }

            dataset["scenes"].append(scene_data)

        if self.enable_logging:
            print(f"[FINALIZE] Dataset complete with {len(dataset['scenes'])} scenes")

        return {"final_dataset": dataset}

    def _should_continue_scenes(self, state: GraphState) -> str:
        """
        Conditional edge: Continue generating scenes or move to next stage.

        Args:
            state: Current graph state

        Returns:
            "continue" to generate more scenes, "next" to proceed
        """
        current = state["current_scene_index"]
        total = state["num_scenes"]

        if current < total:
            return "continue"
        else:
            return "next"

    def run(self, profile: str, num_scenes: int = 5) -> Dict[str, Any]:
        """
        Run the dataset generation pipeline.

        Args:
            profile: Person's profile description
            num_scenes: Number of scenes to generate

        Returns:
            Complete dataset
        """
        if self.enable_logging:
            print(f"\n{'='*60}")
            print(f"DATASET GENERATION PIPELINE")
            print(f"{'='*60}")
            print(f"Profile: {profile[:100]}...")
            print(f"Scenes to generate: {num_scenes}")
            print(f"{'='*60}\n")

        # Initialize state
        initial_state = {
            "profile": profile,
            "num_scenes": num_scenes,
            "current_scene_index": 0,
            "scenes": [],
            "device_data": [],
            "calendar_events": [],
            "messages": [],
            "deficiencies": [],
            "final_dataset": {},
            "error": None
        }

        # Run graph
        result = self.graph.invoke(initial_state)

        return result.get("final_dataset", result)
