import time
from typing import Optional, Tuple
from dataclasses import dataclass

import webrtcvad
import numpy as np
import scipy.signal  # For proper resampling

from config import config

@dataclass
class VadState:
    """State container for Voice Activity Detection processing."""
    is_speech: bool = False
    speech_start_time: Optional[float] = None
    speech_probability: float = 0.0
    consecutive_speech: int = 0
    consecutive_silence: int = 0
    should_clear_buffer: bool = False
    pause_detected: bool = False  # Indicates a natural pause/sentence boundary
    last_speech_time: Optional[float] = None
    current_segment_duration: float = 0.0

class VadProcessor:
    """Processes audio frames to detect voice activity using WebRTC VAD."""
    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_LEVEL) if config.VAD_ENABLED else None
        self.state = VadState()
        self.frame_buffer = []
        self.buffer_size = 2
        self.last_debug_time = time.time()
        self.min_pause_duration = 0.5  # Minimum pause to consider a sentence boundary
        self.max_segment_duration = 5.0  # Maximum duration before forcing a segment break
    
    def process_frame(self, audio_data: bytes, timestamp: float) -> Tuple[bool, float, bool, bool]:
        """Process audio frame with sentence-level VAD detection.
        
        Args:
            audio_data: Raw audio data bytes
            timestamp: Current frame timestamp
            
        Returns:
            Tuple of (is_speech, speech_probability, should_clear_buffer, pause_detected)
        """
        try:
            # If VAD is disabled, treat all audio as speech
            if not config.VAD_ENABLED:
                return True, 1.0, False, False

            self.state.should_clear_buffer = False
            self.state.pause_detected = False
            
            # Proper resampling from original sample rate (assumed 48000Hz) to 16kHz using resample_poly
            samples = np.frombuffer(audio_data, dtype=np.int16)
            # Using resample_poly for proper downsampling with anti-alias filtering
            resampled = scipy.signal.resample_poly(samples, up=1, down=3)
            resampled = resampled.astype(np.int16)
            
            # Get VAD decision
            is_speech = self.vad.is_speech(resampled.tobytes(), 16000)

            # Update buffer
            self.frame_buffer.append(is_speech)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            # Calculate speech ratio
            speech_ratio = sum(self.frame_buffer) / len(self.frame_buffer)
            
            # Update state
            if speech_ratio > 0.5:
                self.state.consecutive_speech += 1
                self.state.consecutive_silence = 0
                self.state.speech_probability = min(1.0, self.state.speech_probability + 0.3)
                self.state.last_speech_time = timestamp
            else:
                self.state.consecutive_speech = 0
                self.state.consecutive_silence += 1
                self.state.speech_probability = max(0.0, self.state.speech_probability - 0.3)
            
            # Handle state transitions
            if not self.state.is_speech:
                if self.state.consecutive_speech >= 2:
                    self.state.is_speech = True
                    self.state.speech_start_time = timestamp
                    self.state.should_clear_buffer = True
                    self.state.current_segment_duration = 0.0
                    print("\n🗣️ Speech started")
            else:
                # Update segment duration
                if self.state.speech_start_time:
                    self.state.current_segment_duration = timestamp - self.state.speech_start_time
                
                # Check for natural pause
                if self.state.last_speech_time and timestamp - self.state.last_speech_time > self.min_pause_duration:
                    self.state.pause_detected = True
                    print("\n⏸️ Natural pause detected")
                
                # Check for maximum segment duration
                if self.state.current_segment_duration > self.max_segment_duration:
                    self.state.pause_detected = True
                    print("\n⏲️ Maximum segment duration reached")
                
                # Handle end of speech
                if self.state.consecutive_silence >= 5:
                    self.state.is_speech = False
                    print("\n🤫 Speech ended")
                    if self.state.speech_start_time:
                        duration = timestamp - self.state.speech_start_time
                        print(f"📊 Speech duration: {duration:.2f}s")
                    self.state.speech_start_time = None
                    self.state.consecutive_speech = 0
                    self.state.should_clear_buffer = True
                    self.state.pause_detected = True
            
            # Debug output every 5 seconds
            current_time = time.time()
            if current_time - self.last_debug_time >= 5.0:
                if self.state.is_speech:
                    duration = timestamp - (self.state.speech_start_time or timestamp)
                    print(f"\n🔍 VAD: Speech ongoing ({duration:.1f}s), "
                          f"prob: {self.state.speech_probability:.2f}")
                else:
                    print(f"\n🔍 VAD: Monitoring, prob: {self.state.speech_probability:.2f}")
                self.last_debug_time = current_time
            
            return (self.state.is_speech, 
                   self.state.speech_probability,
                   self.state.should_clear_buffer,
                   self.state.pause_detected)
            
        except (ValueError, RuntimeError) as e:
            print(f"\n❌ VAD Error: {str(e)}")
            return False, 0.0, False, False
    
    def should_process(self) -> bool:
        """Determine if we should process the current state."""
        # If VAD is disabled, always process
        if not config.VAD_ENABLED:
            return True
            
        if self.state.is_speech:
            return True
        if self.state.consecutive_silence < 3:
            return True
        if self.state.speech_probability > 0.3:
            return True
        return False