import os
import asyncio
import json
import logging
import sys
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from PIL import Image

# [NEW] Import subprocess, sys, and time for starting the Ollama service
import subprocess
import time

import google.generativeai as genai
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.llms import Ollama
import aiohttp


# Custom exception classes
class RAGSystemError(Exception): pass


class EmbeddingError(RAGSystemError): pass


class DatabaseError(RAGSystemError): pass


class LLMError(RAGSystemError): pass


class HealthCheckError(RAGSystemError): pass


@dataclass
class Config:
    """Configuration with hardcoded Gemini keys."""
    db_directory: str = "./chroma_db"
    log_file: str = "rag_api.log"
    embedding_model: str = "all-minilm"
    llm_model: str = "gemini-2.0-flash"
    fallback_llm_model: str = "llama3.2"

    gemini_api_keys: List[str] = field(default_factory=lambda: [
        "AIzaSyCVbypOA1wDnWTkkXzwE0dWzeROmeqvy-I",  # Key 1
        "AIzaSyByZ295lBkZD_4NUivQhR4Octhp9-1f18c",  # Key 2
        "AIzaSyDeteo60vnBzlKfidllx4PfMd4KdSvV_o4",  # Key 3
        "AIzaSyAggYwSznwZWfdR4gX_ZPF7AyEfuU01Itc",  # Key 4
        "AIzaSyADqbidLAZHZ4ZggAeVo_VJ1i2hKwjH7Ew",  # Key 5
        "AIzaSyCXjEXuBhUogZYUAh3vp1Vb2plc-Ir53to",  # Key 6
        "AIzaSyDxFDhSns6_jNnKskUlZTdcdI8oeuLjmMY",  # Key 7
        "AIzaSyAkkjFlgggSRPJWblwbmYCDpH9ujP3jsPI",  # Key 8
        "AIzaSyDWNQv0YXZxD7RuFUBVJAbTIXzctVIfSlg",  # Key 9
        "AIzaSyDTHZHBfmJzVk6IAzAqlxXsXNnuB_2_eRo",  # Key 10
    ])

    # Other config parameters
    chunk_size: int = 1500
    chunk_overlap: int = 200
    similarity_k: int = 6
    mmr_k: int = 6
    mmr_fetch_k: int = 12
    mmr_lambda: float = 0.6
    final_context_chunks: int = 3
    context_score_threshold: Optional[float] = 1.0
    health_check_timeout: int = 15
    internet_check_host: str = "8.8.8.8"
    internet_check_port: int = 53
    internet_check_timeout: int = 3
    has_internet_access: Optional[bool] = field(default=None, init=False)


class HealthChecker:
    @staticmethod
    async def check_internet_connection(host: str, port: int, timeout: int) -> bool:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, OSError):
            return False

    @staticmethod
    async def check_ollama_service(timeout: int = 10) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
                async with s.get('http://localhost:11434/api/tags') as r: return r.status == 200
        except Exception:
            return False

    # [NEW] This new method starts the Ollama service if it's not running
    @staticmethod
    async def start_ollama_service_if_needed():
        """Checks for the Ollama service and starts it if not responsive."""
        logging.info("Checking if Ollama service is running...")
        if await HealthChecker.check_ollama_service(timeout=2):
            logging.info("Ollama service is already running.")
            return

        logging.warning("Ollama service is not responsive. Attempting to start it...")
        try:
            # Use platform-specific flags to run the process in the background without a console window
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW
                # On Windows, `ollama serve` is the command.
                proc = subprocess.Popen(["ollama", "serve"], creationflags=creationflags, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
            else:  # For macOS, Linux, etc.
                # `ollama serve` is the command. It backgrounds itself.
                proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            logging.info(
                f"Launched 'ollama serve' in the background (PID: {proc.pid}). Waiting for it to initialize...")

            # Wait for a few seconds and check again
            for i in range(15):  # Wait up to 15 seconds
                await asyncio.sleep(1)
                if await HealthChecker.check_ollama_service(timeout=2):
                    logging.info("Ollama service started successfully.")
                    return

            # If it's still not running, raise an error
            raise HealthCheckError("Failed to start the Ollama service after multiple attempts.")

        except FileNotFoundError:
            logging.critical("'ollama' command not found. Please ensure Ollama is installed and in your system's PATH.")
            raise HealthCheckError("'ollama' command not found. Is Ollama installed?")
        except Exception as e:
            logging.critical(f"An unexpected error occurred while trying to start Ollama: {e}")
            raise HealthCheckError(f"Failed to start Ollama: {e}")

    @staticmethod
    async def check_embedding_model(model: str, timeout: int = 10) -> bool:
        try:
            em = OllamaEmbeddings(model=model)
            await asyncio.to_thread(em.embed_query, "test")
            return True
        except Exception:
            return False

    @staticmethod
    def check_database_directory(db_path: str) -> bool:
        try:
            p = Path(db_path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "t.tmp").write_text("t")
            (p / "t.tmp").unlink()
            return True
        except Exception:
            return False


class EnhancedRAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.vectorstore: Optional[Chroma] = None
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.llm = None
        self.use_gemini: bool = False
        self.is_initialized: bool = False

    async def initialize(self):
        if self.is_initialized: return
        logging.info("Initializing RAG System...")
        await self.run_health_checks()
        await self.initialize_components()
        self.is_initialized = True
        logging.info("RAG System initialization complete.")

    # [MODIFIED] This method now calls the new function to start Ollama first.
    async def run_health_checks(self):
        logging.info("Running health checks...")

        # [NEW] Start Ollama if it's not running
        await HealthChecker.start_ollama_service_if_needed()

        self.config.has_internet_access = await HealthChecker.check_internet_connection(
            self.config.internet_check_host, self.config.internet_check_port, self.config.internet_check_timeout)
        if not self.config.has_internet_access:
            logging.warning("No internet connection. Gemini LLM will not be available.")
        if not HealthChecker.check_database_directory(self.config.db_directory):
            raise HealthCheckError(f"Database directory {self.config.db_directory} is not accessible.")

        # This check is now more likely to pass
        if not await HealthChecker.check_ollama_service(self.config.health_check_timeout):
            raise HealthCheckError("Ollama service is not responsive. Critical failure.")

        if not await HealthChecker.check_embedding_model(self.config.embedding_model, self.config.health_check_timeout):
            raise HealthCheckError(f"Ollama embedding model '{self.config.embedding_model}' is not available.")

    async def _initialize_gemini_llm(self, api_key: str):
        genai.configure(api_key=api_key)
        llm_candidate = genai.GenerativeModel(self.config.llm_model)
        # Test with a simple text prompt first
        response = await asyncio.to_thread(llm_candidate.generate_content, "Hello", stream=False)
        if not response.text: raise LLMError("Empty response from Gemini during init test.")
        self.llm = llm_candidate
        self.use_gemini = True
        logging.info(f"Gemini LLM initialized successfully using one of the provided API keys.")

    async def initialize_components(self):
        try:
            self.embeddings = OllamaEmbeddings(model=self.config.embedding_model)
            await asyncio.to_thread(self.embeddings.embed_query, "test embedding init")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Ollama embeddings: {e}")

        try:
            self.vectorstore = Chroma(persist_directory=self.config.db_directory, embedding_function=self.embeddings)
        except Exception as e:
            raise DatabaseError(f"Failed to initialize ChromaDB: {e}")

        gemini_initialized = False
        if self.config.has_internet_access and self.config.gemini_api_keys:
            logging.info(
                f"Attempting to initialize Gemini LLM ({self.config.llm_model}) by trying up to {len(self.config.gemini_api_keys)} keys...")
            for i, api_key in enumerate(self.config.gemini_api_keys):
                logging.info(f"--> Trying Gemini API key #{i + 1}...")
                try:
                    await asyncio.wait_for(self._initialize_gemini_llm(api_key),
                                           timeout=self.config.health_check_timeout)
                    gemini_initialized = True
                    break
                except Exception as e:
                    logging.warning(f"Gemini API key #{i + 1} failed: {e}")

            if not gemini_initialized:
                logging.warning("All Gemini API keys failed or timed out.")

        if not gemini_initialized:
            try:
                logging.warning("Falling back to local Ollama LLM.")
                self.llm = Ollama(model=self.config.fallback_llm_model)
                await asyncio.to_thread(self.llm.invoke, "Hello")
                self.use_gemini = False
                logging.info(f"Ollama LLM ({self.config.fallback_llm_model}) initialized as fallback.")
            except Exception as e:
                raise LLMError(f"CRITICAL: All LLMs (Gemini and Ollama) failed to initialize. Last error: {e}")

    def _get_doc_hash(self, doc_content: str) -> str:
        return hashlib.md5(doc_content[:500].encode('utf-8', 'replace')).hexdigest()

    def _rank_and_deduplicate_documents(self, similarity_docs: List[Document], mmr_docs: List[Document]) -> List[Dict]:
        candidates, seen_hashes = [], set()
        for i, doc in enumerate(similarity_docs):
            h = self._get_doc_hash(doc.page_content)
            if h not in seen_hashes:
                candidates.append({'doc': doc, 'score': (self.config.similarity_k - i) * 1.0, 'type': 'sim'})
                seen_hashes.add(h)
        for i, doc in enumerate(mmr_docs):
            h = self._get_doc_hash(doc.page_content)
            if h not in seen_hashes:
                candidates.append({'doc': doc, 'score': (self.config.mmr_k - i) * 0.75, 'type': 'mmr'})
                seen_hashes.add(h)
        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    async def retrieve_context(self, query: str) -> Tuple[str, List[Dict]]:
        if not self.vectorstore: raise RAGSystemError("Vectorstore not initialized.")
        try:
            sim_task = asyncio.to_thread(self.vectorstore.similarity_search, query, k=self.config.similarity_k)
            mmr_task = asyncio.to_thread(self.vectorstore.max_marginal_relevance_search, query, k=self.config.mmr_k,
                                         fetch_k=self.config.mmr_fetch_k)
            sim_docs, mmr_docs = await asyncio.gather(sim_task, mmr_task)
            ranked_candidates = self._rank_and_deduplicate_documents(sim_docs, mmr_docs)
            top_docs = ranked_candidates[:self.config.final_context_chunks]
            context_parts, metadata = [], []
            for item in top_docs:
                if self.config.context_score_threshold is None or item['score'] >= self.config.context_score_threshold:
                    doc = item['doc']
                    context_parts.append(f"Content Block:\n{doc.page_content}")
                    metadata.append({'source': doc.metadata.get('source', 'N/A'), 'score': round(item['score'], 2),
                                     'preview': doc.page_content[:150].replace('\n', ' ') + "..."})
            if not context_parts: return "No relevant context found.", []
            return "\n\n---\n\n".join(context_parts), metadata
        except Exception as e:
            logging.error(f"Error during context retrieval: {e}", exc_info=True)
            raise RAGSystemError(f"Failed to retrieve context: {e}")

    async def generate_response(self, query: str, context: str, image: Optional[Image.Image] = None) -> str:
        if not self.llm: raise LLMError("LLM not initialized.")

        prompt = f"""
        Answer the QUESTION based on:
        1. The provided Content Block(s) (text information)
        2. The provided image (if available)

        If the answer is not in the content or image, say so. 
        Do not mention the words 'context' or 'documents'.

        QUESTION: {query}

        Content Block(s):
        {context}

        ANSWER:
        """

        try:
            if self.use_gemini:
                if image:
                    # For multimodal input (image + text)
                    response = await asyncio.to_thread(
                        self.llm.generate_content,
                        [prompt, image],
                        stream=False
                    )
                else:
                    # For text-only input
                    response = await asyncio.to_thread(
                        self.llm.generate_content,
                        prompt,
                        stream=False
                    )

                if not response.text and hasattr(response, 'prompt_feedback'):
                    return f"[Error: Blocked by safety filters. Reason: {response.prompt_feedback.block_reason}]"
                return response.text.strip()
            else:
                # Fallback for Ollama - can't handle images in this implementation
                if image:
                    logging.warning("Image provided but using Ollama which doesn't support images. Ignoring image.")
                return await asyncio.to_thread(self.llm.invoke, prompt)
        except Exception as e:
            logging.error(f"Error generating LLM response: {e}", exc_info=True)
            raise LLMError(f"Failed to generate response: {e}")

    async def answer_query(self, query: str, image: Optional[Image.Image] = None) -> Dict:
        if not self.is_initialized:
            raise RAGSystemError("RAG System not initialized.")
        logging.info(f"Processing query: '{query[:100]}...'")

        context, doc_info = await self.retrieve_context(query)

        # Handle case where no documents are found
        if not doc_info and not image:
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "links": []
            }

        # Generate the LLM's answer using the retrieved context and/or image
        response = await self.generate_response(query, context, image)

        # Reformat the source document information into the desired "links" structure
        links = [
            {
                "url": info.get('source', 'N/A'),
                "text": info.get('preview', '')
            }
            for info in doc_info
        ]

        return {"answer": response, "links": links}