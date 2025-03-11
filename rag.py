# rag.py
import os
import time
from typing import List, Dict, Any, Optional, Union, Generator, Iterator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output import Generation

# Load environment variables
load_dotenv()

# Check required environment variables
required_env_vars = ["OPENAI_API_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

class HarryPotterRAG:
    def __init__(self):
        """Initialize the RAG system for Harry Potter content."""
        self.groq_client = ChatGroq(temperature=0, model_name="llama3-70b-8192", streaming=True)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=4)  # For handling async tasks

    def load_book(self, file_path: Optional[str] = None, cache_path: str = "hp_vectorstore.faiss", use_cache: bool = True) -> None:
        """Load the existing vector store or create a new one if necessary."""
        if use_cache:
            try:
                print(f"Loading vector store from: {cache_path}")
                self.vector_store = FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)
                print("Vector store loaded successfully")
                self._setup_qa_chain()
                self.is_initialized = True
                return
            except Exception as e:
                print(f"Error loading vector store: {e}")
        
        if file_path:
            print(f"Creating new vector store from: {file_path}")
            self._create_new_vector_store(file_path, cache_path)
        else:
            raise ValueError("No existing vector store found and no file path provided for creation")

    def _create_new_vector_store(self, file_path: str, cache_path: str) -> None:
        """Create a new vector store from PDF file."""
        print(f"Loading PDF file: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        print(f"PDF has {len(pages)} pages")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        all_chunks = text_splitter.split_documents(pages)
        print(f"Split PDF into {len(all_chunks)} chunks")
        
        # Create vector store in batches
        if len(all_chunks) > 500:
            print("Large document detected, creating vector store in batches")
            first_batch = all_chunks[:500]
            self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
            
            for i in range(500, len(all_chunks), 500):
                end_idx = min(i + 500, len(all_chunks))
                print(f"Processing chunks {i} to {end_idx}")
                batch = all_chunks[i:end_idx]
                self.vector_store.add_documents(batch)
        else:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        
        print("Vector store created successfully")
        self.vector_store.save_local(cache_path)
        print(f"Vector store saved to: {cache_path}")
        
        self._setup_qa_chain()
        self.is_initialized = True

    def _setup_qa_chain(self) -> None:
        """Set up the QA chain with Groq."""
        # Use existing groq_client instead of creating a new one
        llm = self.groq_client
        
        prompt_template = """You are an assistant that provides factual information about the Harry Potter books.

Guidelines for generating answers:
- Provide direct, factual answers based solely on the content from the Harry Potter books.
- Focus on accurately answering questions with specific information from the text.
- Do not roleplay as a character or respond as if you are part of the story.
- Cite specific details and events from the books when relevant.
- If information isn't available in the context, simply state that you don't have that information.

Context: {context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
        print("QA chain initialized with Groq")

    def ask(self, question: str) -> str:
        """
        Ask a question about Harry Potter.
        
        Args:
            question: The question to ask about Harry Potter
            
        Returns:
            A string response (collected from streaming output)
        """
        if not self.is_initialized:
            return "Please initialize the Harry Potter knowledge base first."
        
        if not question or len(question.strip()) == 0:
            return "I didn't catch that. Could you please ask your question again?"
        
        try:
            print(f"Querying RAG with: '{question}'")
            # Get streaming generator
            response_gen = self._stream_rag_response(question)
            # Collect all chunks into a single string
            return "".join(list(response_gen))
        except Exception as e:
            print(f"Error querying Groq with RAG: {e}")
            
            try:
                print("Falling back to direct query")
                # Get streaming generator from direct_query
                direct_gen = self.direct_query(question)
                # Collect all chunks into a single string
                return "".join(list(direct_gen))
            except Exception as e2:
                print(f"Error with fallback query: {e2}")
                return "I'm having trouble answering that right now. Could you try rephrasing your question?"

    def _stream_rag_response(self, question: str) -> Iterator[str]:
        """Stream responses from the RAG chain."""
        try:
            # Get the retrieved documents first
            docs = self.vector_store.similarity_search(question, k=4)
            
            if not docs:
                # If no relevant documents found, fall back to direct query
                print("No relevant documents found, falling back to direct query")
                return self.direct_query(question)
                
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Use the template manually instead of through the chain
            prompt_template = """You are an assistant that provides factual information about the Harry Potter books.

Guidelines for generating answers:
- Provide direct, factual answers based solely on the content from the Harry Potter books.
- Focus on accurately answering questions with specific information from the text.
- Do not roleplay as a character or respond as if you are part of the story.
- Cite specific details and events from the books when relevant.
- If information isn't available in the context, simply state that you don't have that information.

Context: {context}

Question: {question}

Answer: """
            
            formatted_prompt = prompt_template.format(context=context, question=question)
            
            # Stream results directly from the LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", formatted_prompt),
                ("user", question)
            ])
            
            chain = prompt | self.groq_client
            # Initialize an empty string to collect response
            full_response = ""
            for chunk in chain.stream({}):
                # Append to full response
                full_response += chunk.content
                # Pass the chunk along
                yield chunk.content
                
        except Exception as e:
            print(f"Error streaming RAG response: {e}")
            yield f"Error processing your question: {str(e)}"

    def direct_query(self, question: str) -> Iterator[str]:
        """Direct query to Groq without RAG for simpler questions."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant that provides factual information about the Harry Potter books.

Guidelines for generating answers:
- Provide direct, factual answers based solely on the content from the Harry Potter books.
- Focus on accurately answering questions with specific information from the text.
- Do not roleplay as a character or respond as if you are part of the story.
- Cite specific details and events from the books when relevant.
- If you don't know the answer, simply state that you don't have that information."""),
                ("user", """Question: {question}""")
            ])
            
            chain = prompt | self.groq_client
            for chunk in chain.stream({"question": question}):
                yield chunk.content
        except Exception as e:
            print(f"Error with direct query: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"

    def stream_response(self, question: str) -> Iterator[str]:
        """
        Streaming version that returns chunks as they're generated.
        This method should only be used when you have a custom way to handle streaming.
        
        Args:
            question: The question to ask about Harry Potter
            
        Returns:
            Iterator yielding response chunks as they're generated
        """
        if not self.is_initialized:
            yield "Please initialize the Harry Potter knowledge base first."
            return
        
        if not question or len(question.strip()) == 0:
            yield "I didn't catch that. Could you please ask your question again?"
            return
        
        try:
            print(f"Streaming RAG response for: '{question}'")
            # Use the streaming generator directly
            for chunk in self._stream_rag_response(question):
                yield chunk
        except Exception as e:
            print(f"Error streaming RAG response: {e}")
            
            try:
                print("Falling back to direct query streaming")
                for chunk in self.direct_query(question):
                    yield chunk
            except Exception as e2:
                print(f"Error with fallback query streaming: {e2}")
                yield "I'm having trouble answering that right now. Could you try rephrasing your question?"