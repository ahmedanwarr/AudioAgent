# app.py
import streamlit as st
import asyncio
from audio_agent import TranscriptCollector, TextToSpeech, get_transcript
from rag import HarryPotterRAG
import os

class AudioRAGApp:
    def __init__(self):
        self.transcript_collector = TranscriptCollector()
        self.tts = TextToSpeech()
        self.rag = HarryPotterRAG()
        
        # Initialize RAG system
        if not hasattr(st.session_state, 'rag_initialized'):
            self.rag.load_book(file_path = "harry-potter-sorcerers-stone.pdf", cache_path="hp_vectorstore.faiss")
            st.session_state.rag_initialized = True
            
        # Initialize conversation state
        if 'conversation_active' not in st.session_state:
            st.session_state.conversation_active = False

    async def run_conversation(self):
        while st.session_state.conversation_active:
            def handle_transcript(transcript):
                st.session_state.current_transcript = transcript
            
            # Reset the transcript
            st.session_state.current_transcript = ""
            
            try:
                # Get audio input and transcribe
                await get_transcript(handle_transcript)
                
                # Get the transcript
                transcript = st.session_state.current_transcript
                
                if transcript:
                    # Create containers for the conversation
                    with st.container():
                        st.write(f"You: {transcript}")
                        
                        # Get RAG response
                        response = self.rag.ask(transcript)
                        
                        # Display the response
                        st.write(f"Assistant: {response}")
                        
                        # Convert response to speech
                        self.tts.speak(response)
                        
                        # Check for exit command
                        if "goodbye" in transcript.lower():
                            st.session_state.conversation_active = False
                            st.write("Conversation ended. Refresh the page to start a new conversation.")
                            break
                
            except Exception as e:
                st.error(f"Error during conversation: {str(e)}")
                st.session_state.conversation_active = False
                break

def main():
    st.title("Harry Potter Audio Question Answering")
    st.write("Start a conversation about Harry Potter!")

    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = AudioRAGApp()
    
    # Create a button to start/stop conversation
    if not st.session_state.conversation_active:
        if st.button("🎤 Start Conversation"):
            st.session_state.conversation_active = True
            st.write("Conversation started! Speak your question...")
            # Run the conversation
            asyncio.run(st.session_state.app.run_conversation())
    else:
        st.write("Conversation is active. Say 'goodbye' to end the conversation.")
        # Display a stop button
        if st.button("⏹️ Stop Conversation"):
            st.session_state.conversation_active = False
            st.write("Conversation ended. Refresh the page to start a new conversation.")
            st.rerun()

if __name__ == "__main__":
    main()