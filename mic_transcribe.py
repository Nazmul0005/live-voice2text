import pyaudio
import wave
import numpy as np
import tempfile
import os
from faster_whisper import WhisperModel
import keyboard
import logging

def get_input_device():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    # Print all audio devices
    print("\nAvailable Audio Input Devices:")
    print("-" * 30)
    for i in range(numdevices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:  # If it's an input device
            print(f"Device {i}: {device_info.get('name')}")
    
    # Get default input device
    default_device = p.get_default_input_device_info()
    print(f"\nUsing default input device: {default_device.get('name')}")
    return default_device.get('index')

def record_and_transcribe():
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # Changed to more compatible format
    CHANNELS = 1
    RATE = 16000
    
    print("Loading Whisper model...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    p = pyaudio.PyAudio()
    
    # Get and print input device info
    input_device_index = get_input_device()
    
    print("\nStarting microphone stream... Press and hold SPACE to record, release to transcribe.")
    print("Press 'q' to quit")
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK
        )
        
        while True:
            frames = []
            audio_data = []
            
            # Wait for SPACE key to be pressed
            keyboard.wait('space', suppress=True)
            print("\nRecording... (Release SPACE to stop)")
            
            # Record while SPACE is held
            while keyboard.is_pressed('space'):
                if keyboard.is_pressed('q'):
                    return
                
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Convert to numpy array for level detection
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    audio_data.extend(audio_array)
                    
                    # Print audio level (simple VU meter)
                    level = np.abs(audio_array).mean()
                    bars = '#' * int(50 * level / 10000)
                    print(f"\rAudio Level: {bars:<50}", end='', flush=True)
                    
                except Exception as e:
                    print(f"\nError reading from stream: {e}")
                    continue
            
            if len(frames) == 0:
                continue
            
            print("\nProcessing...")
            
            # Check if we actually got audio
            audio_data = np.array(audio_data)
            if len(audio_data) > 0:
                max_amplitude = np.abs(audio_data).max()
                print(f"Max audio amplitude: {max_amplitude}")
                if max_amplitude < 100:  # Threshold for silence
                    print("Warning: Very low audio level detected. Please speak louder or check your microphone.")
            
            # Save the recorded data as a WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_filename = temp_file.name
            
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            try:
                # Transcribe the temporary file
                segments, info = model.transcribe(temp_filename, beam_size=1)
                segments = list(segments)
                
                # Print transcription
                if segments:
                    print("\nTranscription:")
                    for segment in segments:
                        print(segment.text)
                else:
                    print("\nNo speech detected - If you were speaking, please check:")
                    print("1. Your microphone is properly connected and selected")
                    print("2. You're speaking loud enough")
                    print("3. Your microphone isn't muted in Windows settings")
            except Exception as e:
                print(f"\nError during transcription: {e}")
            
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            if keyboard.is_pressed('q'):
                break
    
    except Exception as e:
        print(f"\nError opening audio stream: {e}")
        print("Please check if your microphone is properly connected and enabled.")
    
    finally:
        # Clean up
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()

if __name__ == "__main__":
    record_and_transcribe()