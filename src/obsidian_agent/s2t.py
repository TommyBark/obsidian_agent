import time

import keyboard
import pyaudio
from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
from six.moves import queue

# Insert your access token here
access_token = "02E93RoGk5RWaBfcUt9K8SfOVmR8I2WYoHbp5V_9FaDfBFjtVij4BBbG328KUxyquq9asn0gk5lv_7SBDRXu5JxzS2qbc"

class MicrophoneStream(object):
    """
    Opens a recording stream as a generator yielding audio chunks.
    Modified to yield data only while the spacebar is held down.
    """
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer.
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """
        Continuously collect data from the audio stream into the buffer.
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """
        Original generator that yields all audio data.
        """
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)

    def controlled_generator(self):
        """
        Generator that yields audio chunks only while the spacebar is held.
        Also prints the timestamp from the first captured audio chunk.
        """
        first_chunk = True
        while not self.closed and keyboard.is_pressed('space'):
            try:
                chunk = self._buff.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                return
            if first_chunk:
                # Print the timestamp of the first captured audio chunk.
                print(time.strftime('%H:%M:%S', time.localtime()))
                first_chunk = False
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)

# Sampling rate of your microphone and desired chunk size
rate = 44100
chunk = int(rate / 10)

# Creates a media config with settings for a raw microphone input.
example_mc = MediaConfig('audio/x-raw', 'interleaved', rate, 'S16LE', 1)

# Create the streaming client.
streamclient = RevAiStreamingClient(access_token, example_mc)

def main():
    print("Hold SPACEBAR to record and transcribe in real time.")
    print("Release SPACEBAR to end the current session.\n")
    while True:
        # Wait until the spacebar is pressed.
        keyboard.wait('space')
        with MicrophoneStream(rate, chunk) as stream:
            try:
                # Start the streaming transcription using the controlled generator.
                response_gen = streamclient.start(stream.controlled_generator())
                for response in response_gen:
                    # Print each response received from Rev.ai.
                    print(response)
            except Exception as e:
                print("Error during streaming:", e)
        # Print separator after the session ends.
        print("-------------")
        # Short delay to avoid accidental retriggering.
        time.sleep(0.2)

if __name__ == '__main__':
    main()
