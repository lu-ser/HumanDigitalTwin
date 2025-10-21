"""
Streamlit UI for Dataset Generation.
Standalone app for generating synthetic life scene datasets.
"""

import streamlit as st
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Life Scene Dataset Generator",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Life Scene Dataset Generator")
st.markdown("Generate synthetic life scene datasets with device data, calendar, messages, and health analysis.")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Load config
config_path = Path(__file__).parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")

if not api_key:
    st.warning("⚠️ Please provide a Groq API key in the sidebar or set GROQ_API_KEY environment variable.")
    st.stop()

# Number of scenes
num_scenes = st.sidebar.slider(
    "Number of Scenes",
    min_value=1,
    max_value=10,
    value=config['generation']['num_scenes'],
    help="How many life scenes to generate"
)

# Model selection
model = st.sidebar.selectbox(
    "LLM Model",
    options=[
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ],
    index=0
)

# Temperature
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=config['llm']['temperature'],
    step=0.1,
    help="Higher = more creative, Lower = more deterministic"
)

# Output format
output_format = st.sidebar.selectbox(
    "Output Format",
    options=["json", "csv", "both"],
    index=0
)

# Enable logging
enable_logging = st.sidebar.checkbox("Enable Logging", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 View Configs")
if st.sidebar.button("Show Device Config"):
    devices_path = Path(__file__).parent / "config" / "devices.yaml"
    with open(devices_path, 'r') as f:
        st.sidebar.json(yaml.safe_load(f))

if st.sidebar.button("Show Deficiencies Config"):
    deficiencies_path = Path(__file__).parent / "config" / "deficiencies.yaml"
    with open(deficiencies_path, 'r') as f:
        st.sidebar.json(yaml.safe_load(f))

# Main content
st.header("👤 Person Profile")
st.markdown("Provide a detailed description of the person whose life scenes you want to generate.")

# Example profiles
example_profiles = {
    "Software Engineer": "Marco is a 32-year-old software engineer living in Milan. He works remotely from home most days, typically starting work at 9 AM and finishing around 6 PM. He enjoys running in the mornings 3 times a week and goes to the gym on weekends. Marco is generally health-conscious but sometimes skips breakfast when he has early meetings. He's sociable, often meeting friends for dinner or drinks on Friday evenings. He struggles with maintaining a consistent sleep schedule, especially when working on tight deadlines.",

    "Medical Student": "Sarah is a 26-year-old medical student in her final year. She has an extremely busy schedule with lectures, hospital rotations, and study sessions. She wakes up at 6 AM most days and often studies until midnight. Physical activity is limited to walking between hospital buildings. She relies heavily on coffee to stay alert and sometimes skips meals. Sarah is stressed but tries to maintain social connections through group study sessions and occasional video calls with family.",

    "Retired Teacher": "Linda is a 68-year-old retired high school teacher who now volunteers at the local library twice a week. She enjoys a relaxed lifestyle with morning walks, gardening, and reading. She maintains regular meal times and has a consistent sleep schedule, typically sleeping from 10 PM to 7 AM. Linda is socially active, participating in a book club and having weekly lunch dates with friends. She uses a smartwatch to monitor her heart rate and activity levels, as recommended by her doctor.",

    "Fitness Influencer": "Alex is a 28-year-old fitness influencer and personal trainer based in Los Angeles. They wake up at 5:30 AM every day for a morning workout, which they film for their social media channels. Alex teaches three fitness classes per week and has personal training sessions with clients throughout the day. They are very conscious about nutrition, tracking macros and meal prep on Sundays. Alex is highly active on Instagram and TikTok, posting daily content and engaging with followers. Despite their healthy lifestyle, they struggle with work-life balance and often experience stress from constantly being 'on' for social media. They have trouble unplugging in the evenings.",

    "Freelance Designer": "Emma is a 35-year-old freelance graphic designer and single mother of a 7-year-old daughter. She works from home, juggling client projects with school runs and childcare. Her schedule is irregular - sometimes working late into the night to meet deadlines, other times taking afternoon breaks to pick up her daughter. Emma tries to maintain a yoga routine three times per week but often skips it when busy. She has a supportive network of other parents she connects with at school and occasional video calls with her sister abroad. Sleep is inconsistent, averaging 6 hours per night, and she relies on coffee throughout the day. Emma uses a smartwatch mainly to track steps and set reminders for picking up her daughter.",

    "Warehouse Worker": "James is a 45-year-old warehouse supervisor working the evening shift (2 PM - 10 PM) at a distribution center. He's been in this role for 15 years and is on his feet most of the day, walking between storage areas and loading docks. James wakes up around 10 AM, has a late breakfast, and goes to the gym for 30 minutes before work. He's trying to lose weight and monitors his activity with a fitness tracker. After work, he unwinds by watching TV or playing video games until 1-2 AM, which disrupts his sleep schedule. James has limited social interaction during the week due to his shift, but bowls in a league on Sunday mornings and has dinner with his brother's family once a week. He struggles with maintaining a healthy diet, often eating fast food during his work break.",

    "Graduate Researcher": "Dr. Yuki Tanaka is a 29-year-old postdoctoral researcher in neuroscience at a university research lab. She typically arrives at the lab by 8 AM and often doesn't leave until 7 or 8 PM, running experiments, analyzing data, and writing papers. Her work is mentally demanding and deadline-driven, especially when preparing for conferences or grant submissions. Physical activity is minimal - mostly walking around campus and occasionally swimming on weekends. Yuki has an irregular eating schedule, sometimes forgetting lunch when absorbed in her work. She maintains close contact with her family in Japan through weekly video calls and has a small circle of fellow researchers she meets for coffee or dinner. Sleep quality varies significantly based on work stress, ranging from 5 to 8 hours per night. She's recently started using meditation apps to manage anxiety.",

    "Small Business Owner": "Carlos is a 52-year-old owner of a family-run Italian restaurant. He wakes up at 6 AM to go to the wholesale market for fresh ingredients, returns to prep the kitchen by 9 AM, and typically closes the restaurant around 11 PM. It's a 6-day work week with Mondays off. Physical activity comes from constant movement in the kitchen - standing, lifting supplies, and rushing during dinner service. Carlos doesn't track his health formally but has been feeling more tired lately. He has strong social connections through regular customers and his tight-knit family who helps run the business. His wife worries about his health and bought him a smartwatch last month. Meals are abundant but irregular - tasting dishes throughout the day rather than sitting down for proper meals. He enjoys his Monday off by sleeping in, having a long lunch with his wife, and taking a walk in the park."
}

# Profile selection
profile_option = st.selectbox(
    "Choose an example or write your own",
    options=["Custom"] + list(example_profiles.keys())
)

if profile_option == "Custom":
    profile = st.text_area(
        "Profile Description",
        height=200,
        placeholder="Enter a detailed description of the person (lifestyle, age, work, habits, health concerns, social life, etc.)"
    )
else:
    profile = st.text_area(
        "Profile Description",
        value=example_profiles[profile_option],
        height=200
    )

# Generate button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    generate_button = st.button("🚀 Generate Dataset", type="primary", use_container_width=True)

# Generation logic
if generate_button:
    if not profile or len(profile.strip()) < 50:
        st.error("❌ Please provide a detailed profile (at least 50 characters)")
    else:
        try:
            # Import here to avoid loading issues
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "src"))

            from agents.dataset_generation_graph import DatasetGenerationGraph
            from utils.exporter import export_to_json, export_to_csv

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔧 Initializing pipeline...")
            progress_bar.progress(10)

            # Initialize graph
            graph = DatasetGenerationGraph(
                llm_api_key=api_key,
                llm_model=model,
                temperature=temperature,
                enable_logging=enable_logging
            )

            status_text.text(f"🎬 Generating {num_scenes} life scenes...")
            progress_bar.progress(30)

            # Generate dataset
            result = graph.run(
                profile=profile,
                num_scenes=num_scenes
            )

            progress_bar.progress(80)
            status_text.text("💾 Exporting dataset...")

            # Export
            output_dir = Path(__file__).parent / "output"
            saved_files = []

            if output_format in ["json", "both"]:
                json_path = export_to_json(result, output_dir=str(output_dir))
                saved_files.append(json_path)

            if output_format in ["csv", "both"]:
                csv_path = export_to_csv(result, output_dir=str(output_dir))
                saved_files.append(csv_path)

            progress_bar.progress(100)
            status_text.text("✅ Dataset generation complete!")

            # Display results
            st.success(f"✅ Dataset generated successfully with {len(result['scenes'])} scenes!")

            # Show saved files
            st.markdown("### 📁 Saved Files")
            for filepath in saved_files:
                st.code(filepath)

            # Display dataset preview
            st.markdown("---")
            st.header("📊 Dataset Preview")

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Scenes", len(result['scenes']))
            with col2:
                total_events = sum(len(scene.get('calendar_events', [])) for scene in result['scenes'])
                st.metric("Calendar Events", total_events)
            with col3:
                total_messages = sum(len(scene.get('messages', [])) for scene in result['scenes'])
                st.metric("Messages", total_messages)
            with col4:
                # Count scenes with deficiencies
                scenes_with_issues = 0
                for scene in result['scenes']:
                    deficiencies = scene.get('deficiencies', {})
                    has_issue = any(
                        v.get('status') in ['WARNING', 'INSUFFICIENT', 'HIGH']
                        for k, v in deficiencies.items()
                        if k != 'scene_id' and isinstance(v, dict)
                    )
                    if has_issue:
                        scenes_with_issues += 1
                st.metric("Scenes w/ Health Issues", scenes_with_issues)

            # Scene tabs
            st.markdown("### 🎬 Scenes")
            scene_tabs = st.tabs([f"Scene {i+1}" for i in range(len(result['scenes']))])

            for i, tab in enumerate(scene_tabs):
                with tab:
                    scene = result['scenes'][i]

                    # Scene info
                    st.markdown(f"**{scene['day']} - {scene['time']}**")

                    # Format description with proper line breaks
                    description = scene['description'].replace('\\n', '\n')
                    st.text_area("Dialogue", description, height=200, disabled=True)

                    # Device data
                    with st.expander("📱 Device Data", expanded=False):
                        device_data = scene.get('device_data', {})
                        # Remove scene_id and error from display, filter empty values
                        if isinstance(device_data, dict):
                            display_data = {k: v for k, v in device_data.items() if k not in ['scene_id', 'error'] and v}
                            if display_data:
                                st.json(display_data)
                            else:
                                st.info("No device data")
                        else:
                            st.info("No device data")

                    # Calendar events
                    with st.expander("📅 Calendar Events"):
                        events = scene.get('calendar_events', [])
                        if events and isinstance(events, list) and len(events) > 0:
                            for event in events:
                                st.markdown(f"- **{event['title']}** ({event['start_time']} - {event['end_time']}) @ {event['location']}")
                        else:
                            st.info("No calendar events")

                    # Messages
                    with st.expander("💬 Messages"):
                        messages = scene.get('messages', [])
                        if messages and isinstance(messages, list) and len(messages) > 0:
                            for msg in messages:
                                direction_icon = "📤" if msg['direction'] == 'sent' else "📥"
                                st.markdown(f"{direction_icon} **{msg['platform']}** - {msg['contact']} ({msg['timestamp']})")
                                st.markdown(f"> {msg['content']}")
                        else:
                            st.info("No messages")

                    # Health deficiencies
                    with st.expander("🏥 Health Analysis"):
                        deficiencies = scene.get('deficiencies', {})
                        if deficiencies:
                            for category, data in deficiencies.items():
                                if category != 'scene_id' and isinstance(data, dict):
                                    status = data.get('status', 'UNKNOWN')
                                    reason = data.get('reason', 'No data')

                                    # Status emoji
                                    if status == 'OK':
                                        emoji = "✅"
                                        color = "green"
                                    elif status in ['WARNING', 'UNKNOWN']:
                                        emoji = "⚠️"
                                        color = "orange"
                                    else:
                                        emoji = "❌"
                                        color = "red"

                                    st.markdown(f"{emoji} **{category.replace('_', ' ').title()}**: {status}")
                                    st.caption(reason)
                        else:
                            st.info("No health analysis")

            # Download button
            st.markdown("---")
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"dataset_{num_scenes}_scenes.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ Error generating dataset: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("### 📖 About")
st.markdown("""
This tool generates synthetic life scene datasets for testing and development of health monitoring systems.
Each scene includes:
- 📝 Textual description of a life moment
- 📱 Plausible device sensor data (smartwatch, GPS, phone, smartring)
- 📅 Calendar events
- 💬 Messaging app data
- 🏥 Health deficiency analysis

The generated data is synthetic and for research/development purposes only.
""")
