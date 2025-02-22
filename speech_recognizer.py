import vosk
import sounddevice as sd
import numpy as np
import json
import time
import os
import queue
import re
import subprocess
from typing import Optional, List
from queue import Queue
from threading import Thread, Event, Lock, Condition
from dataclasses import dataclass

from config import config
from audio_processor import AudioProcessor
from vad_processor import VadProcessor
from api_client import ApiClient

@dataclass
class AudioChunk:
    data: bytes
    timestamp: float
    is_speech: bool = False
    frame_id: int = 0
    speech_prob: float = 0.0

class SpeechRecognizer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.vad_processor = VadProcessor()
        self.api_client = ApiClient()
        
        # Initialize Vosk model
        print(f"\n🔄 Loading Vosk model from: {config.MODEL_PATH}")
        if not os.path.exists(config.MODEL_PATH):
            raise Exception(f"Model not found: {config.MODEL_PATH}")
        
        self.model = vosk.Model(config.MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(self.model, config.SAMPLE_RATE)
        print("✅ Speech recognition initialized")
        
        # Queues
        self.audio_queue = Queue(maxsize=10)  # Smaller queue for lower latency
        self.transcription_queue = Queue(maxsize=5)
        self.response_queue = Queue(maxsize=5)
        self.tts_queue = Queue()
        
        # Threading controls
        self.interrupt_event = Event()
        self.shutdown_event = Event()
        self.audio_lock = Lock()
        self.tts_lock = Lock()
        self.processing_condition = Condition()
        
        # State
        self.is_speaking = False
        self.is_speech = False
        self.last_processed_text = ""
        self.current_tts_process: Optional[subprocess.Popen] = None
        self.last_frame_time = 0
        self.frame_counter = 0
        
    def start(self):
        """Start all processing threads."""
        self.threads = {
            "audio": Thread(target=self._audio_thread, daemon=True),
            "transcription": Thread(target=self._transcription_thread, daemon=True),
            "tts": Thread(target=self._tts_thread, daemon=True)
        }
        
        for thread in self.threads.values():
            thread.start()
        
        # Initialize audio stream
        device_id = self.audio_processor.select_input_device()
        if device_id is None:
            raise Exception("No input device available")
        
        # Optimize stream configuration
        device_info = sd.query_devices(device_id)
        suggested_latency = device_info.get('default_low_input_latency', 0.01)
        blocksize = int(config.SAMPLE_RATE * 0.01)  # 10ms blocks for better timing
        channels = min(device_info.get('max_input_channels', 1), 2)  # Use mono or stereo
        
        print(f"\n🔧 Audio Configuration:")
        print(f"Sample Rate: {config.SAMPLE_RATE}Hz")
        print(f"Block Size: {blocksize} samples ({blocksize/config.SAMPLE_RATE*1000:.1f}ms)")
        print(f"Latency: {suggested_latency*1000:.1f}ms")
        print(f"Channels: {channels}")
        
        self.stream = sd.InputStream(
            device=device_id,
            samplerate=config.SAMPLE_RATE,
            callback=self._audio_callback,
            dtype=np.float32,
            blocksize=blocksize,
            channels=channels,
            latency=suggested_latency
        )
        
        with self.stream:
            print("\n✨ Ready for speech input...")
            try:
                while not self.shutdown_event.is_set():
                    time.sleep(config.THREAD_SLEEP_TIME)
            except KeyboardInterrupt:
                self.stop()
    
    def stop(self):
        """Stop all processing and cleanup."""
        print("\n🛑 Shutting down...")
        self.shutdown_event.set()
        self.interrupt_event.set()
        
        if self.current_tts_process:
            self.current_tts_process.terminate()
        
        for thread in self.threads.values():
            thread.join(timeout=2.0)
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio data."""
        if status:
            print(f"⚠️ Audio Status: {status}")
        
        current_time = time.time()
        frame_interval = current_time - self.last_frame_time
        
        # Skip frame if it's too close to the previous one
        if frame_interval < 0.01:  # Minimum 10ms between frames
            return
        
        processed_data, level = self.audio_processor.process_format(indata)
        if processed_data and not self.audio_queue.full():
            self.frame_counter += 1
            self.audio_queue.put(AudioChunk(
                data=processed_data,
                timestamp=current_time,
                frame_id=self.frame_counter
            ), block=False)
            self.last_frame_time = current_time
        else:
            # Skip frame if queue is full to prevent backlog
            self.frame_counter += 1
            
            with self.processing_condition:
                self.processing_condition.notify()
    
    def _audio_thread(self):
        """Process audio frames with VAD."""
        last_frame_id = 0
        while not self.shutdown_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.005)  # 5ms timeout
                if chunk is None:
                    continue
                
                # Process with VAD
                is_speech, speech_prob = self.vad_processor.process_frame(chunk.data, chunk.timestamp)
                self.is_speech = is_speech
                
                chunk.is_speech = is_speech
                chunk.speech_prob = speech_prob
                
                if self.vad_processor.should_process() and chunk.frame_id > last_frame_id:
                    last_frame_id = chunk.frame_id
                    self.transcription_queue.put(chunk)
                
            except queue.Empty:
                continue
    
    def _transcription_thread(self):
        """Handle speech recognition and transcription."""
        buffer = []
        last_process_time = time.time()
        max_buffer_frames = int(config.MAX_BUFFER_SECONDS * 50)  # ~50fps
        last_processed_frame_id = 0
        
        while not self.shutdown_event.is_set():
            try:                
                chunk = self.transcription_queue.get(timeout=0.01)  # 10ms timeout
                if chunk is None:
                    continue
                
                # Always append audio data to maintain context
                if chunk.frame_id > last_processed_frame_id:
                    buffer.append(chunk.data)
                    
                    # Process buffer in these cases:
                    # 1. Active speech with enough audio
                    # 2. End of speech segment
                    # 3. Buffer getting too large
                    current_time = time.time()
                    buffer_duration = len(buffer) * 0.02  # ~20ms per frame
                    
                    should_process = (
                        # Regular processing during speech
                        ((chunk.is_speech or chunk.speech_prob > config.SILENCE_THRESHOLD_PROB) and
                         buffer_duration >= 0.3 and  # At least 300ms of audio
                         current_time - last_process_time >= 0.2) or  # Process every 200ms
                        
                        # Process on definite speech end
                        (not chunk.is_speech and
                         chunk.speech_prob < config.SILENCE_THRESHOLD_PROB and
                         buffer_duration >= 0.5) or  # Ensure we have enough context
                        
                        # Process if buffer is getting too large
                        len(buffer) > max_buffer_frames
                    )
                    
                    if should_process:
                        result = self._process_buffer(buffer[-max_buffer_frames:])
                        if result:
                            self.response_queue.put(result)
                        
                        last_processed_frame_id = chunk.frame_id
                        last_process_time = current_time
                        
                        # Keep more context for better word completion
                        overlap_frames = int(0.3 * 50)  # 300ms overlap
                        if len(buffer) > max_buffer_frames:
                            buffer = buffer[-overlap_frames:]
                
            except queue.Empty:
                continue
    
    def _process_buffer(self, buffer: List[bytes]) -> Optional[str]:
        """Process audio buffer and return transcription."""
        if not buffer:
            return None
        
        try:
            # Join latest audio data
            audio_data = b''.join(buffer)
            
            # Try to get a complete result first
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                
                if text and text != self.last_processed_text:
                    print(f"\n📝 Transcribed: {text}")
                    # Reset recognizer to prevent accumulation
                    self.recognizer = vosk.KaldiRecognizer(self.model, config.SAMPLE_RATE)
                    self.last_processed_text = text
                    return text
            else:
                # If no complete result, check partial
                partial = json.loads(self.recognizer.PartialResult())
                if partial.get("partial"):
                    partial_text = partial["partial"].strip()
                    if partial_text and len(partial_text.split()) >= 2:
                        # Only show partial if it's different
                        if partial_text != self.last_processed_text:
                            print(f"\n✏️ Partial: {partial_text}")
                            self.last_processed_text = partial_text
                        return partial_text
            
            return self.last_processed_text if self.is_speech else None
        
        except Exception as e:
            print(f"\n❌ Recognition Error: {str(e)}")
        
        return None
    
    def _tts_thread(self):
        """Handle text-to-speech generation and playback."""
        while not self.shutdown_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text is None:
                    continue
                
                if not self.interrupt_event.is_set():
                    with self.tts_lock:
                        # Process API response
                        response = self.api_client.process_response(text)
                        if response:
                            # Split into sentences for streaming
                            sentences = re.split(r'(?<=[.!?])\s+', response)
                            for sentence in sentences:
                                if sentence.strip():
                                    self._generate_speech(sentence.strip())
                                    time.sleep(0.1)  # Small delay between sentences
                            
                            # Save conversation
                            self.api_client.save_conversation(text, response)
                
            except queue.Empty:
                continue
    
    def _generate_speech(self, text: str) -> None:
        """Generate and play speech for the given text."""
        if not text:
            return
        
        try:
            with self.tts_lock:
                # Kill any existing TTS process
                if self.current_tts_process and self.current_tts_process.poll() is None:
                    self.current_tts_process.terminate()
                    self.current_tts_process.wait()
                
                # Get voice setting
                voice = os.getenv('SYNTHIA_VOICE', config.DEFAULT_VOICE)
                print(f"\n🔊 Speaking: {text}")
                
                # Create output directory
                os.makedirs('output', exist_ok=True)
                output_file = f"output/speech_{int(time.time() * 1000)}.mp3"
                
                try:
                    # Generate speech
                    edge_tts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               ".venv", "bin", "edge-tts")
                    
                    self.current_tts_process = subprocess.Popen([
                        edge_tts_path,
                        '--voice', voice,
                        '--rate', '-10%',
                        '--text', text,
                        '--write-media', output_file
                    ])
                    
                    self.current_tts_process.wait()
                    
                    if self.current_tts_process.returncode == 0:
                        # Play audio
                        self.is_speaking = True
                        self.current_tts_process = subprocess.Popen(
                            ['afplay', '-q', '1', '-r', '1.0', output_file]
                        )
                        self.current_tts_process.wait()
                        self.is_speaking = False
                        
                        # Cleanup
                        if os.path.exists(output_file):
                            os.remove(output_file)
                    else:
                        raise Exception("TTS generation failed")
                
                except Exception as e:
                    print(f"⚠️ Edge TTS failed, falling back to say: {e}")
                    self.current_tts_process = subprocess.Popen(
                        ['say', '-v', voice, '-r', '175', text]
                    )
                    self.current_tts_process.wait()
        
        except Exception as e:
            print(f"❌ TTS Error: {str(e)}")
            try:
                subprocess.run(['say', text], check=True)
            except Exception as e:
                print(f"❌ Fallback TTS Error: {str(e)}")
        finally:
            self.current_tts_process = None
            self.is_speaking = False

def main():
    """Main entry point."""
    try:
        recognizer = SpeechRecognizer()
        recognizer.start()
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
    finally:
        if 'recognizer' in locals():
            recognizer.stop()

if __name__ == "__main__":
    main()