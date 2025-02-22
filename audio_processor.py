import numpy as np
import sounddevice as sd
import time
import os
from typing import Optional, Tuple
from config import config

class AudioProcessor:
    def __init__(self):
        self.level_buffer = []
        self.max_level = 0.1  # Dynamic max level tracking
        self.frame_counter = 0
        self.last_frame_time = 0
        self.frame_buffer = []
        self.last_frame_hash = None
        self.min_frame_interval = 1.0 / 50  # 20ms for 50fps target
        
    def process_format(self, indata: np.ndarray) -> Tuple[Optional[bytes], float]:
        """Process audio format with improved quality and level visualization."""
        try:
            # Convert float32 to int16 and handle stereo
            if len(indata.shape) > 1 and indata.shape[1] > 1:
                mono_data = np.mean(indata, axis=1)
            else:
                mono_data = indata.flatten()

            # Check frame timing
            current_time = time.time()
            if current_time - self.last_frame_time < self.min_frame_interval:
                return None, 0.0
            
            # Check for duplicate frames
            frame_hash = hash(mono_data.tobytes())
            if frame_hash == self.last_frame_hash:
                return None, 0.0
            self.last_frame_hash = frame_hash
            
            # Add to frame buffer
            self.frame_buffer.append(mono_data)
            
            # Keep buffer size limited
            frame_size = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
            max_frames = int(config.MAX_BUFFER_SECONDS * (1000 / config.VAD_FRAME_MS))
            if len(self.frame_buffer) > max_frames:
                self.frame_buffer = self.frame_buffer[-max_frames:]
            
            self.last_frame_time = current_time
            mono_data = np.concatenate(self.frame_buffer[-1:])  # Use latest frame
            
            # Calculate RMS level with smoothing
            rms = np.sqrt(np.mean(np.square(mono_data)))
            self.level_buffer.append(rms)
            if len(self.level_buffer) > 10:  # Keep last 10 samples for smoothing
                self.level_buffer.pop(0)
            
            # Update max level with decay
            self.max_level = max(np.mean(self.level_buffer), self.max_level * 0.95)
            
            # Normalize level relative to max
            normalized_level = rms / (self.max_level if self.max_level > 0 else 1.0)
            
            # Enhanced visualization with gradient
            bar_length = int(normalized_level * 30)  # 30 characters max
            if bar_length > 30:
                bar_length = 30
            
            # Color gradient based on level
            if bar_length <= 10:
                color = "\033[32m"  # Green
            elif bar_length <= 20:
                color = "\033[33m"  # Yellow
            else:
                color = "\033[31m"  # Red
            
            level_bar = "█" * bar_length
            spaces = " " * (30 - bar_length)
            print(f"\r🎙️ Level: {color}{level_bar}{spaces}\033[0m", end="", flush=True)
            
            # Convert to 16-bit PCM with proper scaling and noise gate
            noise_gate = 0.001  # -60dB

            # Apply noise gate with soft knee
            gate_curve = np.clip(np.abs(mono_data) / noise_gate, 0, 1)
            gate_curve = np.power(gate_curve, 2)  # Quadratic curve for smoother transition
            gated_data = mono_data * gate_curve

            self.frame_counter += 1

            # Add small amount of dither before quantization
            dither = np.random.uniform(-1, 1, len(gated_data)) * 1e-6
            dithered_data = gated_data + dither
            
            int16_data = (dithered_data * 32767).astype(np.int16)
            return int16_data[:frame_size].tobytes(), normalized_level
            return None, normalized_level
            
        except Exception as e:
            print(f"\n❌ Audio Processing Error: {str(e)}")
            return None, 0.0
    
    @staticmethod
    def list_devices() -> list:
        """List available audio devices with improved formatting."""
        devices = []
        try:
            device_list = sd.query_devices()
            print("\n🎤 Available Audio Devices:")
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    info = f"[{i}] {device['name']}"
                    info += f"\n    Channels: {device['max_input_channels']}"
                    info += f"\n    Sample Rate: {device['default_samplerate']:.0f}Hz"
                    print(info)
                    devices.append(device)
            return devices
        except Exception as e:
            print(f"❌ Error listing audio devices: {str(e)}")
            return []
    
    def select_input_device(self) -> Optional[int]:
        """Select audio input device with improved detection."""
        try:
            devices = self.list_devices()
            if not devices:
                print("❌ No audio input devices found!")
                return None
            
            # Check for environment variable
            custom_device = os.getenv('SYNTHIA_MIC')
            if custom_device:
                for i, device in enumerate(devices):
                    if custom_device.lower() in device['name'].lower():
                        print(f"🎙️ Using configured device: {device['name']}")
                        return i
            
            # Look for common input device keywords
            keywords = ['mic', 'input', 'headset', 'audio in']
            for i, device in enumerate(devices):
                name = device['name'].lower()
                if any(keyword in name for keyword in keywords):
                    print(f"🎙️ Auto-selected device: {device['name']}")
                    return i
            
            # Fallback to first input device
            print(f"🎙️ Using default device: {devices[0]['name']}")
            return 0
            
        except Exception as e:
            print(f"❌ Error selecting input device: {str(e)}")
            return None