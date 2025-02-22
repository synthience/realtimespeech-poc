
# 🚀 Real-Time AI Speech Pipeline: Synthia with Local Speech Processing

# **📌 Installation Commands**
# Ensure all dependencies are installed before running the script.
# pip install vosk sounddevice soundfile webrtcvad numpy requests edge-tts scipy

import sounddevice as sd
import soundfile as sf
import queue
import json
import os
import sys
import time
import requests
import webrtcvad
import subprocess
import numpy as np
import threading
import re
import select
from threading import Thread, Event, Lock, Condition
from queue import Queue
from dataclasses import dataclass
from typing import Optional, List, Dict
import scipy.signal  # Added for proper resampling
from config import config  # Import config module

# Ensure vosk is installed
try:
    import vosk
except ImportError:
    print("❌ Vosk not found. Installing required dependencies...")
    subprocess.check_call(["pip", "install", "vosk", "sounddevice", "soundfile", "webrtcvad", "numpy", "requests", "scipy"])
    import vosk
# **📌 Edge TTS Configuration**
EDGE_TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "edge-tts")
EDGE_TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "edge-tts")

# **📌 Thread Management**
@dataclass
class AudioChunk:
    data: bytes
    timestamp: float
    is_speech: bool = False

class ThreadManager:
    def __init__(self):
        self.audio_queue = Queue()
        self.vad_queue = Queue()
        self.transcription_queue = Queue()
        self.response_queue = Queue()
        self.tts_queue = Queue()
        
        self.interrupt_event = Event()
        self.shutdown_event = Event()
        self.audio_lock = Lock()
        self.tts_lock = Lock()
        self.processing_condition = Condition()
        
        self.active_threads: Dict[str, Thread] = {}
        self.current_tts_process: Optional[subprocess.Popen] = None
        self.is_speaking = False
        self.last_processed_text = ""
        
    def start_threads(self):
        thread_configs = {
            "audio_processor": (self.audio_processing_thread, ()),
            "vad_processor": (self.vad_processing_thread, ()),
            "transcription": (self.transcription_thread, ()),
            "api_communication": (self.api_communication_thread, ()),
            "tts_generator": (self.tts_generation_thread, ()),
            "playback": (self.audio_playback_thread, ())
        }
        
        for name, (target, args) in thread_configs.items():
            thread = Thread(target=target, args=args, name=name, daemon=True)
            thread.start()
            self.active_threads[name] = thread
            
    def stop_threads(self):
        self.shutdown_event.set()
        self.interrupt_event.set()
        
        for thread in self.active_threads.values():
            thread.join(timeout=2.0)
            
        if self.current_tts_process:
            self.current_tts_process.terminate()
            
    def audio_processing_thread(self):
        buffer = []
        samples_needed = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)  # samples per frame
        
        while not self.shutdown_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    continue
                
                # Add to buffer
                buffer.append(chunk)
                total_samples = sum(len(np.frombuffer(b, dtype=np.int16)) for b in buffer)
                
                # Process complete frames
                while total_samples >= samples_needed:
                    # Combine buffer chunks
                    all_samples = np.concatenate([np.frombuffer(b, dtype=np.int16) for b in buffer])
                    
                    # Extract complete frame
                    frame_data = all_samples[:samples_needed].tobytes()
                    
                    # Keep remaining samples in buffer
                    remaining_samples = all_samples[samples_needed:]
                    buffer = [remaining_samples.tobytes()] if len(remaining_samples) > 0 else []
                    total_samples = len(remaining_samples)
                    
                    # Send frame to VAD
                    with self.audio_lock:
                        self.vad_queue.put(AudioChunk(
                            data=frame_data,
                            timestamp=time.time()
                        ))
                    
            except queue.Empty:
                continue
                
    def vad_processing_thread(self):
        consecutive_speech_frames = 0
        consecutive_silence_frames = 0
        last_debug_time = time.time()
        is_currently_speech = False
        speech_probability = 0.0  # Track speech probability
        
        while not self.shutdown_event.is_set():
            try:
                chunk = self.vad_queue.get(timeout=0.1)
                if chunk is None:
                    continue
                
                # If VAD is disabled, pass all audio directly to transcription
                if not config.VAD_ENABLED:
                    chunk.is_speech = True
                    self.transcription_queue.put(chunk)
                    continue
                
                # Debug VAD every 5 seconds
                current_time = time.time()
                if current_time - last_debug_time >= 5.0:
                    print(f"\n🔍 VAD active, speech prob: {speech_probability:.2f}")
                    last_debug_time = current_time
                
                try:
                    # Proper resampling from 48kHz to 16kHz using resample_poly
                    samples = np.frombuffer(chunk.data, dtype=np.int16)
                    resampled = scipy.signal.resample_poly(samples, up=1, down=3).astype(np.int16)
                    
                    # VAD expects 16-bit PCM at 16kHz
                    is_speech = vad.is_speech(resampled.tobytes(), 16000)
                    
                    # Update speech probability with stronger hysteresis
                    if is_speech:
                        consecutive_speech_frames += 1
                        consecutive_silence_frames = 0
                        # Faster ramp up for speech detection
                        speech_probability = min(1.0, speech_probability + 0.3)
                    else:
                        consecutive_speech_frames = 0
                        consecutive_silence_frames += 1
                        # Slower ramp down for better phrase completion
                        speech_probability = max(0.0, speech_probability - 0.05)
                    
                    # State transitions with clear thresholds
                    if not is_currently_speech and speech_probability > config.SPEECH_THRESHOLD:
                        is_currently_speech = True
                        if self.is_speaking:
                            self.interrupt_event.set()
                        print("\n🗣️ Speech started")
                    elif is_currently_speech and speech_probability < config.SILENCE_THRESHOLD_PROB:
                        is_currently_speech = False
                        print("\n🤫 Speech ended")
                        # Process buffer when speech ends with complete silence
                        if consecutive_silence_frames >= config.MIN_SPEECH_FRAMES:
                            self.transcription_queue.put(AudioChunk(
                                data=chunk.data,
                                timestamp=time.time(),
                                is_speech=False
                            ))
                    
                    # Set chunk speech flag and pass to transcription
                    chunk.is_speech = is_currently_speech
                    if is_currently_speech or speech_probability > config.SILENCE_THRESHOLD_PROB:
                        self.transcription_queue.put(chunk)
                        
                except Exception as e:
                    print(f"\n❌ VAD Error: {str(e)}")
                    print(f"Data length: {len(chunk.data)}, Expected: {config.CHUNK_SIZE * 2}")
                    continue
                    
            except queue.Empty:
                continue
                
    def transcription_thread(self):
        buffer = []
        last_speech_time = time.time()
        last_debug_time = time.time()
        last_process_time = time.time()
        min_chunk_duration = 0.3  # 300ms minimum for chunk processing
        max_chunk_duration = 2.0  # 2s maximum before forced processing
        min_samples = 48000 * 0.3  # Minimum samples for processing (300ms)
        
        while not self.shutdown_event.is_set():
            try:
                chunk = self.transcription_queue.get(timeout=0.1)
                if chunk is None:
                    continue
                
                current_time = time.time()
                
                # Debug transcription every 5 seconds
                if current_time - last_debug_time >= 5.0:
                    total_samples = sum(len(np.frombuffer(b, dtype=np.int16)) for b in buffer)
                    print(f"\n🔍 Transcription active, buffer size: {total_samples} samples")
                    last_debug_time = current_time
                
                # In non-VAD mode, only process when we have data and recording is active
                if not config.VAD_ENABLED:
                    if chunk.data:  # Real audio data
                        if recording:
                            buffer.append(chunk.data)
                    else:  # Control signal
                        if chunk.is_speech:  # Start recording
                            buffer = []  # Clear buffer
                        else:  # Stop recording, process buffer
                            if buffer:
                                print("\n🎯 Processing recorded audio...")
                                self._process_buffer(buffer)
                                buffer = []
                                # Force final processing
                                if hasattr(self, 'accumulated_text') and self.accumulated_text:
                                    print("\n✅ Processing final context...")
                                    combined_text = ' '.join(self.accumulated_text)
                                    if combined_text:
                                        self.response_queue.put(combined_text)
                                    self.accumulated_text = []
                                # Reset recognizer state after sending
                                recognizer.Reset()
                else:
                    # Normal VAD mode processing
                    if chunk.is_speech:
                        buffer.append(chunk.data)
                        last_speech_time = chunk.timestamp
                        
                        # Calculate buffer duration
                        total_samples = sum(len(np.frombuffer(b, dtype=np.int16)) for b in buffer)
                        buffer_duration = total_samples / config.SAMPLE_RATE
                        
                        # Process buffer if:
                        # 1. We have enough samples and enough time has passed
                        # 2. Or if we've hit the maximum duration
                        current_time = time.time()
                        if ((total_samples >= min_samples and
                             current_time - last_process_time >= min_chunk_duration) or
                            buffer_duration >= max_chunk_duration):
                            
                            print(f"\n🎯 Processing chunk ({buffer_duration:.1f}s)...")
                            self._process_buffer(buffer)
                            last_process_time = current_time
                            # Keep accumulating text, don't reset state
                            buffer = []  # Clear audio buffer only
                    else:
                        # If we have buffered speech and hit silence, process it
                        if buffer:
                            total_samples = sum(len(np.frombuffer(b, dtype=np.int16)) for b in buffer)
                            if total_samples >= min_samples:
                                print("\n🎯 Processing on silence...")
                                self._process_buffer(buffer)
                                buffer = []
                                last_process_time = time.time()
                                # Don't reset recognizer to maintain context
                                # Signal end of speech for response timing
                                if hasattr(self, 'accumulated_text') and self.accumulated_text:
                                    print("\n🤫 Speech segment complete")
                                    # Force process any remaining context
                                    combined_text = ' '.join(self.accumulated_text)
                                    if combined_text:
                                        print("\n✅ Processing final context...")
                                        self.response_queue.put(combined_text)
                                        self.accumulated_text = []
                    
            except queue.Empty:
                # Process any remaining buffer periodically
                if buffer:
                    current_time = time.time()
                    if current_time - last_process_time >= max_chunk_duration:
                        total_samples = sum(len(np.frombuffer(b, dtype=np.int16)) for b in buffer)
                        if total_samples >= min_samples:
                            print("\n⚠️ Processing stale buffer...")
                            self._process_buffer(buffer)
                            buffer = []
                            last_process_time = current_time
                continue
                
    def _process_buffer(self, buffer):
        """Process accumulated audio buffer and manage transcription context."""
        if not buffer:
            return
            
        try:
            # Combine all chunks
            audio_data = b''.join(buffer)
            
            # Process with Vosk
            recognizer.AcceptWaveform(audio_data)
            
            # Get result without resetting state
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            
            if text:
                print(f"\n📝 Transcribed: {text}")
                print("\r" + " " * 30 + "\r", end="")  # Clear audio level bar
                
                # Accumulate text until we have complete sentences
                if not hasattr(self, 'accumulated_text'):
                    self.accumulated_text = []
                self.accumulated_text.append(text)
                
                # Check if we have enough context (e.g., multiple sentences or long enough)
                combined_text = ' '.join(self.accumulated_text)
                sentence_count = len(re.findall(r'[.!?]+', combined_text))
                
                if sentence_count >= 2 or len(combined_text.split()) >= 20:
                    print("\n✅ Processing complete context...")
                    self.response_queue.put(combined_text)
                    self.accumulated_text = []  # Reset accumulation
                    # Don't reset recognizer state to maintain context
                else:
                    print("\n📝 Accumulating context...")
                    
        except Exception as e:
            print(f"\n❌ Transcription Error: {str(e)}")

    def api_communication_thread(self):
        response_buffer = []  # Buffer for queued responses
        last_speech_time = time.time()
        speech_timeout = 2.0  # Time to wait after speech ends before playing responses
        
        while not self.shutdown_event.is_set():
            try:
                text = self.response_queue.get(timeout=0.1)
                if text is None:
                    continue
                    
                # Handle voice change command
                if "change voice to" in text.lower():
                    new_voice = text.lower().replace("change voice to", "").strip()
                    if new_voice in available_voices:
                        os.environ['SYNTHIA_VOICE'] = new_voice
                        print(f"🎤 Changed voice to: {new_voice}")
                        response_buffer.append("Voice changed successfully")
                    else:
                        response_buffer.append("Voice not found. Please use a valid voice name or number.")
                elif "use voice" in text.lower():
                    try:
                        # Get voice number
                        number = int(text.lower().replace("use voice", "").strip())
                        if 1 <= number <= len(available_voices):
                            new_voice = available_voices[number - 1]
                            os.environ['SYNTHIA_VOICE'] = new_voice
                            print(f"🎤 Changed voice to: {new_voice}")
                            response_buffer.append("Voice changed successfully")
                        else:
                            response_buffer.append("Invalid voice number. Please choose a number between 1 and " + str(len(available_voices)))
                    except (ValueError, IndexError):
                        response_buffer.append("Invalid voice number. Please use a valid number.")
                else:
                    # Process normal response
                    response = process_synthia_response(text)
                    if response:
                        # Split response into sentences and add to buffer
                        sentences = re.split(r'(?<=[.!?])\s+', response)
                        response_buffer.extend([s.strip() for s in sentences if s.strip()])
                        print(f"\n💭 Queued {len(sentences)} sentences for response")
                
                # Check if we should play responses
                current_time = time.time()
                if response_buffer and current_time - last_speech_time >= speech_timeout:
                    print("\n🗣️ Playing queued responses...")
                    for sentence in response_buffer:
                        self.tts_queue.put(sentence)
                        time.sleep(0.1)  # Small delay between sentences
                    response_buffer = []  # Clear the buffer after playing
                
            except queue.Empty:
                # Update last speech time when we're actively receiving audio
                if hasattr(self, 'accumulated_text') and self.accumulated_text:
                    last_speech_time = time.time()
                continue
                
    def tts_generation_thread(self):
        while not self.shutdown_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text is None:
                    continue
                    
                if not self.interrupt_event.is_set():
                    with self.tts_lock:
                        synthesize_speech(text)
            except queue.Empty:
                continue
                
    def audio_playback_thread(self):
        while not self.shutdown_event.is_set():
            with self.processing_condition:
                self.processing_condition.wait()
                if self.interrupt_event.is_set():
                    if self.current_tts_process:
                        self.current_tts_process.terminate()
                    self.interrupt_event.clear()
                    self.tts_queue.queue.clear()

# **📌 Available Edge TTS Voices**
available_voices = []

def load_available_voices():
    """Load and cache available Edge TTS voices."""
    global available_voices
    try:
        result = subprocess.run([EDGE_TTS_PATH, '--list-voices'], capture_output=True, text=True)
        if result.returncode == 0:
            available_voices = [v.split('\t')[0] for v in result.stdout.split('\n') if v.strip()]
            print("\n🎤 Available Edge TTS Voices:")
            for i, voice in enumerate(available_voices, 1):
                print(f"{i}. {voice}")
    except Exception as e:
        print(f"❌ Error loading voices: {str(e)}")

# Initialize thread manager
thread_manager = ThreadManager()

# **📌 Initialize Vosk for Real-Time Speech Recognition**
print("\n🔄 Loading Vosk model from:", config.MODEL_PATH)
try:
    if not os.path.exists(config.MODEL_PATH):
        print("❌ Error: Vosk model directory not found!")
        print("Please ensure you have downloaded the model and extracted it to:", config.MODEL_PATH)
        sys.exit(1)
        
    vosk_model = vosk.Model(config.MODEL_PATH)
    print("✅ Vosk model loaded successfully")
    
    recognizer = vosk.KaldiRecognizer(vosk_model, config.SAMPLE_RATE)  # Use configured sample rate
    print("✅ Speech recognizer initialized")
except Exception as e:
    print("❌ Error initializing Vosk:", str(e))
    sys.exit(1)

# **📌 Initialize Voice Activity Detection (VAD) for Interruptions**
vad = webrtcvad.Vad(3)  # Sensitivity level 3

def list_audio_devices():
    """Lists available audio devices."""
    try:
        devices = sd.query_devices()
        print("\n🎤 Available Audio Devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']} (Inputs: {device['max_input_channels']})")
        return devices
    except Exception as e:
        print(f"❌ Error listing audio devices: {str(e)}")
        return []

def select_input_device():
    """Selects the audio input device."""
    try:
        devices = list_audio_devices()
        if not devices:
            print("❌ No audio devices found!")
            return None

        # Check for environment variable
        custom_device = os.getenv('SYNTHIA_MIC')
        if custom_device:
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and custom_device.lower() in device['name'].lower():
                    print(f"🎙️ Using custom device: {device['name']}")
                    return i

        # Find first input device
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"🎙️ Using default device: {device['name']}")
                return i

        return None
    except Exception as e:
        print(f"❌ Error selecting input device: {str(e)}")
        return None

def process_audio_format(indata):
    """Process audio format for better quality."""
    try:
        # Convert float32 to int16 and handle stereo
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            # Average stereo channels
            mono_data = np.mean(indata, axis=1)
        else:
            mono_data = indata.flatten()
            
        # Show audio level with color
        rms = np.sqrt(np.mean(np.square(mono_data)))
        level = int(rms * 50)  # Scale to 0-50 range
        bar_length = min(level, 20)  # Max 20 chars
        
        # Color coding: green (0-7), yellow (8-14), red (15-20)
        if bar_length <= 7:
            color = "\033[32m"  # Green
        elif bar_length <= 14:
            color = "\033[33m"  # Yellow
        else:
            color = "\033[31m"  # Red
            
        level_bar = "█" * bar_length
        print(f"\r🎙️ Level: {color}{level_bar}\033[0m{' ' * (20 - bar_length)}", end="", flush=True)
            
        # Convert to 16-bit PCM with proper scaling
        int16_data = (mono_data * 32768).astype(np.int16)
        
        # Debug audio format
        if len(int16_data) != config.CHUNK_SIZE:
            print(f"\n⚠️ Unexpected chunk size: {len(int16_data)} samples")
            
        # Ensure we're returning properly formatted bytes for VAD
        # VAD expects 16-bit PCM at 48kHz in 10, 20, or 30ms frames
        # At 48kHz, 20ms = 960 samples
        frame_size = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)  # samples per frame
        if len(int16_data) >= frame_size:
            return int16_data[:frame_size].tobytes()
        else:
            print(f"\n⚠️ Insufficient samples for VAD frame: {len(int16_data)}/{frame_size}")
            return None
    except (AttributeError, ValueError, TypeError) as e:
        print(f"❌ Audio Processing Error: {str(e)}")
        return None

def audio_callback(indata, frames, time, status):
    """Enhanced callback for sounddevice to process audio data."""
    try:
        if status:
            print(f"⚠️ Audio Status: {status}")
        
        processed_data = process_audio_format(indata)
        if processed_data:
            # Only process audio if VAD is enabled or we're recording
            if config.VAD_ENABLED or recording:
                thread_manager.audio_queue.put(processed_data)
                
                # Notify processing condition
                with thread_manager.processing_condition:
                    thread_manager.processing_condition.notify()
    except Exception as e:
        print(f"❌ Audio Processing Error: {str(e)}")

def process_synthia_response(user_message):
    """Sends user speech to Synthia and retrieves a response."""
    if not user_message:
        print("⚠️ Empty user message")
        return None
        
    try:
        headers = {"X-API-KEY": config.API_KEY, "Content-Type": "application/json"}
        payload = {"user_message": user_message}
        
        response = requests.post(config.API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return data.get("synthia_response")
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {str(e)}")
        return None

def synthesize_speech(text):
    """Optimized TTS with process management and streaming."""
    if not text:
        return
        
    try:
        with thread_manager.tts_lock:
            # Kill any existing TTS process
            if thread_manager.current_tts_process and thread_manager.current_tts_process.poll() is None:
                thread_manager.current_tts_process.terminate()
                thread_manager.current_tts_process.wait()
            
            # Get custom voice setting
            voice = os.getenv('SYNTHIA_VOICE', config.DEFAULT_VOICE)
            print(f"🔊 Speaking: {text}")
            
            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)
            output_file = f"output/speech_{int(time.time() * 1000)}.mp3"
            
            try:
                # Generate speech with edge-tts
                thread_manager.current_tts_process = subprocess.Popen([
                    EDGE_TTS_PATH,
                    '--voice', voice,
                    '--rate', '-10%',
                    '--pitch', '+0Hz',  # Natural pitch
                    '--volume', '+0%',  # Full volume
                    '--text', text,
                    '--write-media', output_file
                ])
                
                # Wait for generation to complete
                thread_manager.current_tts_process.wait()
                
                if thread_manager.current_tts_process.returncode == 0:
                    # Set speaking flag
                    thread_manager.is_speaking = True
                    
                    # Play audio with buffer size optimization
                    thread_manager.current_tts_process = subprocess.Popen(['afplay', '-q', '1', '-r', '1.0', output_file])
                    thread_manager.current_tts_process.wait()
                    
                    # Reset speaking flag
                    thread_manager.is_speaking = False
                    
                    # Cleanup
                    if os.path.exists(output_file):
                        os.remove(output_file)
                else:
                    raise Exception("Edge TTS failed to generate audio")
                    
            except Exception as e:
                print(f"⚠️ Edge TTS failed, falling back to say command: {e}")
                # Fallback to optimized say command
                thread_manager.current_tts_process = subprocess.Popen(['say', '-v', voice, '-r', '175', text])
                thread_manager.current_tts_process.wait()
                
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        # Ultimate fallback with basic settings
        try:
            subprocess.run(['say', text], check=True)
        except Exception as e:
            print(f"❌ Fallback TTS Error: {str(e)}")
    finally:
        thread_manager.current_tts_process = None
        thread_manager.is_speaking = False

def save_memory(user_input, ai_response):
    """Stores user conversations in local memory."""
    try:
        with open(config.MEMORY_FILE, "r") as file:
            memory = json.load(file)
    except FileNotFoundError:
        memory = {"history": []}

    memory["history"].append({"user": user_input, "synthia": ai_response})

    with open(config.MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

def load_memory():
    """Loads the last stored conversation."""
    try:
        with open(config.MEMORY_FILE, "r") as file:
            memory = json.load(file)
            return memory["history"][-1] if memory["history"] else {}
    except FileNotFoundError:
        return {}

# **📌 Cleanup and Shutdown**
import signal
import atexit

active_processes = set()
is_shutting_down = False

def register_process(process):
    """Register a process for cleanup."""
    active_processes.add(process)

def cleanup_process(process):
    """Remove process from tracking set."""
    active_processes.discard(process)

def graceful_shutdown(signum=None, frame=None):
    """Handle graceful shutdown of all resources."""
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True
    
    print("\n\n🛑 Shutting down Synthia...")
    
    try:
        # Kill all active processes
        for process in active_processes:
            try:
                process.kill()
            except:
                pass
        
        # Cleanup resources
        if recognizer:
            del recognizer
        if vosk_model:
            del vosk_model
        
        # Remove temporary files
        output_dir = './output'
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                try:
                    os.unlink(os.path.join(output_dir, file))
                except:
                    pass
        
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"❌ Error during shutdown: {str(e)}")
    
    # Force exit
    os._exit(0)

# Register cleanup handlers
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
atexit.register(graceful_shutdown)

# **📌 Initialize System**
print("\n🎤 Edge TTS Voice Settings")
print(f"Current voice: {os.getenv('SYNTHIA_VOICE', config.DEFAULT_VOICE)}")
print("To change voice, say: 'change voice to [name]' or 'use voice [number]'")
print("Example: 'change voice to en-US-AvaNeural' or 'use voice 1'")

# Load available voices
load_available_voices()

print("\n🎙️ Microphone Selection")
print("Available microphones:")
list_audio_devices()
print("To set input device: export SYNTHIA_MIC='device name'")
print("To enable/disable VAD: export SYNTHIA_VAD_ENABLED=false")
print("\nPress CTRL+C to exit")

print("\n✨ Synthia is ready! Listening for input...\n")

# Global recording state
recording = False

def run_cli_mode():
    """Run the system in CLI mode"""
    global recording

    def is_key_pressed():
        # Check if there's input available
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r:
            # Read and clear the input
            sys.stdin.readline()
            return True
        return False

    try:
        # Start all processing threads
        thread_manager.start_threads()
        
        # Initialize audio stream
        device_id = select_input_device()
        if device_id is None:
            raise Exception("No input device available")

        stream = sd.InputStream(
            device=device_id,
            channels=config.CHANNELS,
            samplerate=config.SAMPLE_RATE,
            callback=audio_callback,
            dtype=np.float32,
            blocksize=config.CHUNK_SIZE
        )

        if not config.VAD_ENABLED:
            print("\n✨ Synthia is ready! Press RETURN to start/stop recording, or CTRL+C to exit...\n")
        else:
            print("\n✨ Synthia is ready! Listening for input...\n")

        with stream:
            while not is_shutting_down:
                try:
                    # Process audio in the background via threads
                    time.sleep(0.1)  # Reduce CPU usage
                    
                    # Check for shutdown signal
                    if thread_manager.shutdown_event.is_set():
                        break

                    # Handle manual recording control when VAD is disabled
                    if not config.VAD_ENABLED and is_key_pressed():
                        recording = not recording
                        if recording:
                            print("\n🔴 Recording started...")
                            # Signal recording start
                            thread_manager.transcription_queue.put(AudioChunk(
                                data=b'',
                                timestamp=time.time(),
                                is_speech=True
                            ))
                        else:
                            print("\n⏹️ Recording stopped, processing...")
                            # Signal recording stop to process any remaining audio
                            thread_manager.transcription_queue.put(AudioChunk(
                                data=b'',
                                timestamp=time.time(),
                                is_speech=False
                            ))
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Main Loop Error: {str(e)}")
                    if is_shutting_down:
                        break
                    continue

    except KeyboardInterrupt:
        print("\n⚡ Caught interrupt signal")
    except Exception as e:
        print(f"❌ Fatal Error: {str(e)}")
    finally:
        print("\n🧹 Cleaning up...")
        thread_manager.stop_threads()
        graceful_shutdown()

if __name__ == "__main__":
    try:
        run_cli_mode()
    except KeyboardInterrupt:
        print("\n⚡ Caught interrupt signal")
    except Exception as e:
        print(f"❌ Fatal Error: {str(e)}")
    finally:
        graceful_shutdown()
