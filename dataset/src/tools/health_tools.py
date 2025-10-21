"""
Health validation tools for deficiency detection.
These tools help validate metrics against healthy ranges.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any
import yaml
from pathlib import Path


# Load deficiencies config
_config_path = Path(__file__).parent.parent.parent / "config" / "deficiencies.yaml"
with open(_config_path, 'r', encoding='utf-8') as f:
    _DEFICIENCIES_CONFIG = yaml.safe_load(f)

# Load devices config
_devices_path = Path(__file__).parent.parent.parent / "config" / "devices.yaml"
with open(_devices_path, 'r', encoding='utf-8') as f:
    _DEVICES_CONFIG = yaml.safe_load(f)


class ValidateStepsInput(BaseModel):
    """Input schema for validate_steps_count."""
    steps: int = Field(description="Number of steps")


@tool(args_schema=ValidateStepsInput)
def validate_steps_count(steps: int) -> str:
    """
    Validate if daily step count meets health recommendations.
    Returns status (OK/WARNING/INSUFFICIENT) and explanation.
    """
    thresholds = _DEFICIENCIES_CONFIG['physical_activity']['thresholds']

    if steps >= thresholds['steps_recommended']:
        return f"OK - {steps} steps meets recommended daily goal ({thresholds['steps_recommended']}+)"
    elif steps >= thresholds['steps_minimum']:
        return f"WARNING - {steps} steps is above minimum ({thresholds['steps_minimum']}) but below recommended ({thresholds['steps_recommended']})"
    else:
        return f"INSUFFICIENT - {steps} steps is below minimum recommended ({thresholds['steps_minimum']})"


class ValidateHeartRateInput(BaseModel):
    """Input schema for validate_heart_rate."""
    heart_rate: int = Field(description="Heart rate in bpm")
    context: str = Field(description="Context: rest, light, moderate, or intense activity")


@tool(args_schema=ValidateHeartRateInput)
def validate_heart_rate(heart_rate: int, context: str) -> str:
    """
    Validate if heart rate is appropriate for the activity context.
    Context can be: rest, light, moderate, intense.
    Returns validation result.
    """
    smartwatch_config = _DEVICES_CONFIG['smartwatch']['metrics']['heart_rate']

    context_ranges = {
        'rest': smartwatch_config['rest_range'],
        'light': smartwatch_config['light_activity_range'],
        'moderate': smartwatch_config['moderate_activity_range'],
        'intense': smartwatch_config['intense_activity_range']
    }

    if context not in context_ranges:
        return f"ERROR - Unknown context '{context}'. Use: rest, light, moderate, or intense"

    min_hr, max_hr = context_ranges[context]

    if min_hr <= heart_rate <= max_hr:
        return f"OK - {heart_rate} bpm is normal for {context} activity (range: {min_hr}-{max_hr})"
    elif heart_rate < min_hr:
        return f"WARNING - {heart_rate} bpm is below typical range for {context} activity ({min_hr}-{max_hr})"
    else:
        return f"WARNING - {heart_rate} bpm is above typical range for {context} activity ({min_hr}-{max_hr})"


class ValidateSleepInput(BaseModel):
    """Input schema for validate_sleep_duration."""
    sleep_minutes: int = Field(description="Sleep duration in minutes")


@tool(args_schema=ValidateSleepInput)
def validate_sleep_duration(sleep_minutes: int) -> str:
    """
    Validate if sleep duration meets health recommendations.
    Returns status and explanation.
    """
    thresholds = _DEFICIENCIES_CONFIG['sleep']['thresholds']

    sleep_hours = sleep_minutes / 60

    if sleep_minutes >= thresholds['duration_optimal_minutes']:
        return f"OK - {sleep_hours:.1f} hours is optimal sleep duration"
    elif sleep_minutes >= thresholds['duration_recommended_minutes']:
        return f"OK - {sleep_hours:.1f} hours meets recommended sleep duration"
    elif sleep_minutes >= thresholds['duration_minimum_minutes']:
        return f"WARNING - {sleep_hours:.1f} hours is above minimum but below recommended (7+ hours)"
    else:
        return f"INSUFFICIENT - {sleep_hours:.1f} hours is below minimum recommended sleep (6+ hours)"


class ValidateActivityMinutesInput(BaseModel):
    """Input schema for validate_active_minutes."""
    active_minutes: int = Field(description="Active minutes")


@tool(args_schema=ValidateActivityMinutesInput)
def validate_active_minutes(active_minutes: int) -> str:
    """
    Validate if daily active minutes meet health recommendations.
    Returns status and explanation.
    """
    thresholds = _DEFICIENCIES_CONFIG['physical_activity']['thresholds']

    if active_minutes >= thresholds['active_minutes_recommended']:
        return f"OK - {active_minutes} active minutes meets recommended daily goal (60+ min)"
    elif active_minutes >= thresholds['active_minutes_minimum']:
        return f"WARNING - {active_minutes} active minutes is above minimum (30) but below recommended (60)"
    else:
        return f"INSUFFICIENT - {active_minutes} active minutes is below minimum recommended (30 min)"


class ValidateHRVInput(BaseModel):
    """Input schema for validate_hrv."""
    hrv: int = Field(description="Heart Rate Variability in ms")


@tool(args_schema=ValidateHRVInput)
def validate_hrv(hrv: int) -> str:
    """
    Validate HRV (Heart Rate Variability) - indicator of stress and recovery.
    HRV is measured in milliseconds. Lower HRV can indicate stress or poor recovery.
    Use this tool to check if heart_rate_variability value from smartring is healthy.
    Returns validation result.
    """
    smartring_config = _DEVICES_CONFIG['smartring']['metrics']['heart_rate_variability']
    stress_threshold = _DEFICIENCIES_CONFIG['stress']['thresholds']['hrv_stress_threshold']

    healthy_min, healthy_max = smartring_config['healthy_range']

    if hrv < stress_threshold:
        return f"HIGH STRESS - HRV {hrv}ms is critically low (below {stress_threshold}ms), indicating high stress or poor recovery"
    elif hrv < healthy_min:
        return f"WARNING - HRV {hrv}ms is below healthy range ({healthy_min}-{healthy_max}ms), may indicate stress"
    elif hrv <= healthy_max:
        return f"OK - HRV {hrv}ms is in healthy range ({healthy_min}-{healthy_max}ms)"
    else:
        return f"OK - HRV {hrv}ms is excellent (above {healthy_max}ms)"


class ValidateBodyTempInput(BaseModel):
    """Input schema for validate_body_temperature."""
    temperature: float = Field(description="Body temperature in Celsius")


@tool(args_schema=ValidateBodyTempInput)
def validate_body_temperature(temperature: float) -> str:
    """
    Validate body temperature against normal range.
    Returns validation result.
    """
    temp_config = _DEVICES_CONFIG['smartring']['metrics']['body_temperature']
    normal_min, normal_max = temp_config['normal_range']

    if temperature < normal_min:
        return f"WARNING - Body temperature {temperature}°C is below normal range ({normal_min}-{normal_max}°C)"
    elif temperature <= normal_max:
        return f"OK - Body temperature {temperature}°C is normal ({normal_min}-{normal_max}°C)"
    else:
        return f"WARNING - Body temperature {temperature}°C is above normal range ({normal_min}-{normal_max}°C), may indicate fever"


def get_health_tools() -> list:
    """
    Returns list of all health validation tools.

    Returns:
        List of Langchain tools for health validation
    """
    return [
        validate_steps_count,
        validate_heart_rate,
        validate_sleep_duration,
        validate_active_minutes,
        validate_hrv,
        validate_body_temperature,
    ]
