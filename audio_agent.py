# audio_agent.py
import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import certifi
import ssl
import tempfile


from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakOptions
)

load_dotenv()


class TextToSpeech:
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.deepgram = DeepgramClient(self.api_key)
        self.current_process = None

    def stop_speaking(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.current_process.wait()

    async def speak(self, text, model="aura-stella-en"):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Generate the audio file
            options = SpeakOptions(model=model)
            text_dict = {"text": text}
            self.deepgram.speak.v("1").save(temp_filename, text_dict, options)

            # Play the audio file using ffplay
            if os.path.exists(temp_filename):
                # Store the process so we can interrupt it later
                self.current_process = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", temp_filename],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Don't wait for the process to complete - let it run in background
            else:
                print(f"Error: {temp_filename} not found.")

        except Exception as e:
            print(f"Exception: {e}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback, tts=None):
    transcription_complete = asyncio.Event()
    
    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        
        # Set the SSL certificate verification at the global level
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # Make sure to pass your Deepgram API key
        deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                # Stop TTS if it's currently playing when user starts speaking
                if tts:
                    tts.stop_speaking()
                    print("User started speaking, stopping current TTS playback")
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return
