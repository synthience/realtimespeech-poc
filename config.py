from dataclasses import dataclass, field, fields
from typing import Dict, Any
import os

@dataclass
class Config:
    # API Configuration
    API_URL: str = field(default="http://127.0.0.1:5001/synthiaquery")
    API_KEY: str = field(default="lucidia")
    API_TIMEOUT: int = field(default=10)
    API_MAX_RETRIES: int = field(default=3)
    
    # Audio Processing
    SAMPLE_RATE: int = field(default=48000)
    CHANNELS: int = field(default=1)
    CHUNK_SIZE: int = field(default=960)  # 20ms at 48kHz for VAD frame alignment
    DTYPE: str = field(default='float32')
    
    # Voice Activity Detection
    VAD_ENABLED: bool = field(default=True)  # Enable/disable VAD
    VAD_FRAME_MS: int = field(default=20)
    VAD_LEVEL: int = field(default=3)  # Highest sensitivity
    SILENCE_THRESHOLD: int = field(default=1200)  # ms - increased for better phrase completion
    MIN_SPEECH_FRAMES: int = field(default=10)  # Increased for more stable detection
    MIN_PHRASE_SAMPLES: int = field(default_factory=lambda: int(48000 * 0.5))  # 0.5 seconds
    SPEECH_THRESHOLD: float = field(default=0.4)  # Lower threshold for earlier speech detection
    SILENCE_THRESHOLD_PROB: float = field(default=0.05)  # Lower threshold to maintain speech state longer
    
    # Speech Recognition
    MODEL_PATH: str = field(default="vosk-model-en-us-0.42-gigaspeech")
    DEFAULT_VOICE: str = field(default="en-US-AvaNeural")
    MEMORY_FILE: str = field(default="synthia_memory.json")
    
    # Buffer Management
    MAX_BUFFER_SECONDS: float = field(default=3.0)  # Increased for better context
    MAX_BUFFER_SAMPLES: int = field(default_factory=lambda: int(48000 * 2.0))
    OVERLAP_MS: int = field(default=50)  # Reduced overlap for faster processing
    OVERLAP_SAMPLES: int = field(default_factory=lambda: int(48000 * 50 / 1000))
    
    # Performance
    THREAD_SLEEP_TIME: float = field(default=0.01)  # Reduced sleep time for better responsiveness
    RESOURCE_CHECK_INTERVAL: int = field(default=5)
    CPU_WARNING_THRESHOLD: int = field(default=80)
    MEMORY_WARNING_THRESHOLD: int = field(default=80)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        env_config = {}
        for field_info in fields(cls):
            env_var = f'SYNTHIA_{field_info.name.upper()}'
            if env_var in os.environ:
                env_val = os.environ[env_var]
                field_type = field_info.type
                try:
                    # Convert environment value to correct type
                    if field_type == int:
                        env_config[field_info.name] = int(env_val)
                    elif field_type == float:
                        env_config[field_info.name] = float(env_val)
                    elif field_type == bool:
                        env_config[field_info.name] = env_val.lower() in ('true', '1', 'yes')
                    else:
                        env_config[field_info.name] = env_val
                except ValueError:
                    print(f"⚠️ Invalid value for {env_var}: {env_val}")
        
        return cls(**env_config)

# Create default config instance
config = Config.from_env()