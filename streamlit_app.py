import streamlit as st
import json
import time
import threading
import queue
import numpy as np
import sys
import os
import sounddevice as sd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtimespeech import (
    ThreadManager, load_available_voices, select_input_device,
    process_synthia_response, audio_callback, synthesize_speech,
    init_system
)

# Initialize global thread manager
thread_manager = None

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False
if 'thread_manager' not in st.session_state:
    st.session_state['thread_manager'] = None
if 'status_queue' not in st.session_state:
    st.session_state['status_queue'] = queue.Queue()
if 'audio_level_queue' not in st.session_state:
    st.session_state['audio_level_queue'] = queue.Queue()
if 'transcription_queue' not in st.session_state:
    st.session_state['transcription_queue'] = queue.Queue()
if 'response_queue' not in st.session_state:
    st.session_state['response_queue'] = queue.Queue()

# Page config
st.set_page_config(
    page_title="Synthia Speech Interface",
    page_icon="🎤",
    layout="wide"
)

# Title and description
st.title("🎤 Synthia Speech Interface")
st.markdown("""
This interface provides real-time monitoring of the speech system, including:
- Audio levels and VAD status
- Live transcriptions
- AI responses
- System status
""")

# Create columns for different components
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Status")
    status_placeholder = st.empty()
    
    st.subheader("Audio Level")
    audio_level_placeholder = st.empty()
    
    st.subheader("Available Voices")
    voices_placeholder = st.empty()

with col2:
    st.subheader("Live Transcription")
    transcription_placeholder = st.empty()
    
    st.subheader("AI Response")
    response_placeholder = st.empty()

def update_interface(status_placeholder, audio_level_placeholder, transcription_placeholder, response_placeholder):
    """Update the Streamlit interface with latest data"""
    while True:
        try:
            if 'status_queue' in st.session_state:
                try:
                    while True:
                        status = st.session_state.status_queue.get_nowait()
                        status_placeholder.info(status)
                except queue.Empty:
                    pass
            
            if 'audio_level_queue' in st.session_state:
                try:
                    while True:
                        level_data = st.session_state.audio_level_queue.get_nowait()
                        level = level_data['level']
                        color = level_data['color']
                        bar = "█" * level
                        audio_level_placeholder.markdown(f"<span style='color: {color}'>{bar}</span>", unsafe_allow_html=True)
                except queue.Empty:
                    pass
            
            if 'transcription_queue' in st.session_state:
                try:
                    while True:
                        transcription = st.session_state.transcription_queue.get_nowait()
                        transcription_placeholder.markdown(f"🗣️ {transcription}")
                except queue.Empty:
                    pass
            
            if 'response_queue' in st.session_state:
                try:
                    while True:
                        response = st.session_state.response_queue.get_nowait()
                        response_placeholder.markdown(f"🤖 {response}")
                except queue.Empty:
                    pass
            
            time.sleep(0.1)  # Prevent excessive CPU usage
            
        except Exception as e:
            st.error(f"Error updating interface: {str(e)}")
            time.sleep(1)

def custom_audio_callback(indata, frames, time_info, status):
    """Modified audio callback that also updates the Streamlit interface"""
    try:
        if status:
            st.session_state.status_queue.put(f"⚠️ Audio Status: {status}")
        
        # Convert float32 to int16 and handle stereo
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            mono_data = np.mean(indata, axis=1)
        else:
            mono_data = indata.flatten()
            
        # Calculate audio level
        rms = np.sqrt(np.mean(np.square(mono_data)))
        level = int(rms * 50)  # Scale to 0-50 range
        bar_length = min(level, 20)  # Max 20 chars
        
        # Color coding
        if bar_length <= 7:
            color = "green"
        elif bar_length <= 14:
            color = "yellow"
        else:
            color = "red"
            
        # Update audio level in interface
        st.session_state.audio_level_queue.put({
            'level': bar_length,
            'color': color
        })
        
        # Process audio data as normal
        if st.session_state.thread_manager:
            processed_data = st.session_state.thread_manager.process_audio_format(indata)
            if processed_data:
                st.session_state.thread_manager.audio_queue.put(processed_data)
                
    except Exception as e:
        st.session_state.status_queue.put(f"❌ Audio Processing Error: {str(e)}")

def initialize_system():
    """Initialize the speech system with Streamlit integration"""
    try:
        global thread_manager
        # Initialize system
        thread_manager = init_system()
        st.session_state.thread_manager = thread_manager
        
        try:
            # Initialize audio stream
            device_id = select_input_device()
            if device_id is None:
                raise Exception("No input device available")

            stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=48000,
                callback=custom_audio_callback,
                dtype=np.float32,
                blocksize=960
            )
            
            st.session_state.audio_stream = stream
            stream.start()
            st.session_state.status_queue.put("✅ Audio stream started")
        except Exception as e:
            st.error(f"Failed to initialize audio stream: {str(e)}")
            return
        
        # Override the response handling to update Streamlit
        def custom_response_handler(text):
            st.session_state.transcription_queue.put(text)
            response = st.session_state.thread_manager.process_synthia_response(text)
            if response:
                st.session_state.response_queue.put(response)
                sentences = response.split('. ')
                for sentence in sentences:
                    if sentence.strip():
                        st.session_state.thread_manager.tts_queue.put(sentence.strip())
                        time.sleep(0.1)
        
        st.session_state.thread_manager.handle_response = custom_response_handler
        
        # Start the thread manager
        st.session_state.thread_manager.start_threads()
        
        # Start the interface update thread
        update_thread = threading.Thread(
            target=update_interface,
            args=(status_placeholder, audio_level_placeholder, transcription_placeholder, response_placeholder),
            daemon=True
        )
        update_thread.start()
        
        # Load available voices
        voices = load_available_voices()
        voices_text = "\n".join([f"{i+1}. {voice}" for i, voice in enumerate(voices)])
        voices_placeholder.text(voices_text)
        
        st.session_state.initialized = True
        st.session_state.status_queue.put("✅ System initialized successfully")
        
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")

# Initialize button
if not st.session_state.initialized:
    if st.button("Initialize System"):
        initialize_system()
else:
    st.success("System is running")
    if st.button("Stop System"):
        if st.session_state.thread_manager:
            st.session_state.thread_manager.stop_threads()
        if hasattr(st.session_state, 'audio_stream'):
            st.session_state.audio_stream.stop()
            st.session_state.audio_stream.close()
        st.session_state.initialized = False
        st.rerun()
