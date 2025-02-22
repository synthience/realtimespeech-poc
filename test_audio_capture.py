import sounddevice as sd
import numpy as np
import time
from speech_recognizer import SpeechRecognizer
from audio_processor import AudioProcessor
from vad_processor import VadProcessor
import queue
from typing import List, Dict
import json

def test_audio_capture(duration: int = 5):
    """Test audio capture and processing for a specified duration."""
    print(f"\n🎤 Starting audio capture test ({duration} seconds)...")
    
    # Initialize components
    audio_processor = AudioProcessor()
    vad_processor = VadProcessor()
    
    # Stats tracking
    stats = {
        'total_frames': 0,
        'processed_frames': 0,
        'speech_frames': 0,
        'duplicate_frames': set(),
        'frame_timestamps': [],
        'buffer_sizes': []
    }
    
    # Audio callback to process frames
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"⚠️ Status: {status}")
        
        timestamp = time.time()
        processed_data, level = audio_processor.process_format(indata)
        
        if processed_data:
            stats['total_frames'] += 1
            frame_hash = hash(processed_data)
            
            # Check for duplicates
            if frame_hash in stats['duplicate_frames']:
                print(f"⚠️ Duplicate frame detected at {timestamp}")
            else:
                stats['duplicate_frames'].add(frame_hash)
            
            # Process with VAD
            is_speech, speech_prob = vad_processor.process_frame(processed_data, timestamp)
            if is_speech:
                stats['speech_frames'] += 1
            
            stats['frame_timestamps'].append(timestamp)
            stats['buffer_sizes'].append(len(processed_data))
            stats['processed_frames'] += 1
    
    # Setup stream
    device_id = audio_processor.select_input_device()
    if device_id is None:
        raise Exception("No input device available")
    
    stream = sd.InputStream(
        device=device_id,
        channels=1,
        samplerate=48000,
        callback=audio_callback,
        dtype=np.float32,
        blocksize=960
    )
    
    try:
        with stream:
            print("🎙️ Recording...")
            time.sleep(duration)
        
        # Analyze results
        print("\n📊 Test Results:")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Processed frames: {stats['processed_frames']}")
        print(f"Speech frames: {stats['speech_frames']}")
        
        # Check frame timing
        if len(stats['frame_timestamps']) > 1:
            intervals = np.diff(stats['frame_timestamps'])
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            print(f"\nFrame Timing:")
            print(f"Average interval: {avg_interval*1000:.2f}ms")
            print(f"Interval std dev: {std_interval*1000:.2f}ms")
        
        # Check buffer consistency
        if stats['buffer_sizes']:
            unique_sizes = set(stats['buffer_sizes'])
            print(f"\nBuffer Sizes:")
            print(f"Unique sizes: {len(unique_sizes)}")
            print(f"Expected size: 1920 bytes")
            if len(unique_sizes) > 1:
                print("⚠️ Warning: Inconsistent buffer sizes detected")
        
        # Final assessment
        has_issues = False
        if len(stats['frame_timestamps']) < duration * 50:  # Expect ~50 frames/sec
            print("\n⚠️ Warning: Low frame rate detected")
            has_issues = True
        if std_interval > 0.005:  # More than 5ms jitter
            print("\n⚠️ Warning: High timing jitter detected")
            has_issues = True
        
        if not has_issues:
            print("\n✅ Audio capture test passed - No significant issues detected")
        
    except Exception as e:
        print(f"\n❌ Test Error: {str(e)}")

if __name__ == "__main__":
    test_audio_capture(5)