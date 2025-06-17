import os
import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Google Generative AI for Gemini
import google.generativeai as genai

# LangChain components for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.llms import Ollama

# Selenium for web scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# Custom exception classes for structured error handling
class RAGSystemError(Exception):
    """Base exception for RAG system-related errors."""
    pass

class ScrapingError(RAGSystemError):
    """Exception specific to web scraping failures."""
    pass

class EmbeddingError(RAGSystemError):
    """Exception specific to embedding model failures."""
    pass

class DatabaseError(RAGSystemError):
    """Exception specific to vector database (ChromaDB) failures."""
    pass

class LLMError(RAGSystemError):
    """Exception specific to Language Model failures."""
    pass

class HealthCheckError(RAGSystemError):
    """Exception specific to critical system health check failures."""
    pass


@dataclass
class Config:
    """
    Centralized configuration for the RAG system.
    This dataclass provides a single point of truth for all configurable parameters,
    making the system easier to manage, test, and adapt.
    """
    url_file: str = "all_content_urls.txt"  # Path to the file containing URLs to scrape
    db_directory: str = "./chroma_db"  # Directory for the ChromaDB vector store persistence
    processed_urls_file: str = field(init=False)  # Path to store processed URLs, set in __post_init__
    log_file: str = "rag_system.log"  # Path for the system log file
    embedding_model: str = "all-minilm"  # Ollama embedding model name
    llm_model: str = "gemini-2.0-flash"  # Primary LLM model (Gemini)
    fallback_llm_model: str = "llama3.2"  # Fallback LLM model (Ollama) if Gemini fails or no internet
    gemini_api_keys: List[str] = field(default_factory=lambda: [  # List of Gemini API keys for rotation/redundancy
        os.getenv("GEMINI_API_KEY_1", "AIzaSyD0CpNpMbp1YPN7tDQyhCXxppfvSthdyeQ"), # Placeholder/Default
        os.getenv("GEMINI_API_KEY_2", "AIzaSyCKAdnWiFXsUXBeD4iNrS_Jqeq86ag5mIQ"),
        os.getenv("GEMINI_API_KEY_3", "AIzaSyC52KpQyrRe1TRUAks9Y6POhIGOOM9MYQU"),
        os.getenv("GEMINI_API_KEY_4", "AIzaSyCZPMKSSNVnSYiXobcWKrC0Gr2RPyB-juQ"),
        os.getenv("GEMINI_API_KEY_5", "AIzaSyAt4M65EnlgTcoROatg9-n_i56Zay-GDn4")
    ])
    max_concurrent_scrapes: int = 3  # Maximum number of URLs to scrape simultaneously
    scrape_timeout: int = 45  # Overall timeout for a single page scrape (including load and content extraction)
    min_content_length: int = 100  # Minimum length of extracted text content to be considered valid
    page_load_timeout: int = 20  # Timeout for Selenium to load a page initially
    chunk_size: int = 1500  # Max characters per document chunk for RAG
    chunk_overlap: int = 200  # Overlap between consecutive chunks for context preservation
    similarity_k: int = 6  # Number of documents to retrieve via similarity search
    mmr_k: int = 6  # Number of documents to retrieve via Maximal Marginal Relevance (MMR)
    mmr_fetch_k: int = 12  # Number of documents to fetch before applying MMR re-ranking
    mmr_lambda: float = 0.6  # Balance between relevance and diversity for MMR (0=diversity, 1=relevance)
    final_context_chunks: int = 3  # Maximum number of top chunks to include in the final LLM prompt context
    context_score_threshold: Optional[float] = 1.0 # Minimum score for a retrieved chunk to be included in context. Set to None to disable.
    health_check_timeout: int = 15  # Timeout for individual health checks (e.g., LLM ping)
    retry_attempts: int = 2  # Number of retry attempts for certain operations (currently not fully implemented across the board)
    retry_delay: float = 1.0  # Delay between retries

    internet_check_host: str = "8.8.8.8"  # Host for internet connectivity check (Google DNS)
    internet_check_port: int = 53  # Port for internet connectivity check (DNS port)
    internet_check_timeout: int = 3  # Timeout for internet connectivity check

    has_internet_access: Optional[bool] = field(default=None, init=False) # Runtime flag for internet access

    def __post_init__(self):
        """
        Post-initialization hook to set derived attributes and perform initial validation.
        Ensures `processed_urls_file` is correctly located within the `db_directory`
        and filters out invalid/placeholder Gemini API keys.
        """
        self.processed_urls_file = os.path.join(self.db_directory, "processed_urls.json")
        # Filter out invalid or placeholder API keys
        self.gemini_api_keys = [
            key for key in self.gemini_api_keys
            if key and not (
                key.startswith(("YOUR_GEMINI_API_KEY_HERE", "#AIzaSy", "AIzaSyC8finEK8KkcR0OfEVfqSZJQH48RvTEwFA"))) # Example placeholder keys
        ]
        if not self.gemini_api_keys:
            logging.warning(
                "No valid GEMINI_API_KEYs provided. Gemini LLM will not be available. "
                "Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. as environment variables or in Config."
            )
        else:
            logging.info(f"Found {len(self.gemini_api_keys)} Gemini API key(s) to attempt.")

class HealthChecker:
    """
    Provides static methods for checking the health and connectivity of various system components.
    This helps in early detection of environment issues before starting main operations.
    """

    @staticmethod
    async def check_internet_connection(host: str, port: int, timeout: int) -> bool:
        """
        Checks for internet connectivity by attempting to open a TCP connection to a well-known host/port.
        This is a fundamental check as many RAG components (Gemini, web scraping) rely on internet access.
        """
        try:
            # asyncio.open_connection is non-blocking and suitable for network checks
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout
            )
            writer.close()
            await writer.wait_closed() # Ensure the connection is properly closed
            logging.debug(f"Internet connectivity check: Successful connection to {host}:{port}.")
            return True
        except (asyncio.TimeoutError, OSError) as e:
            # TimeoutError for connection timeout, OSError for general network issues (e.g., host unreachable)
            logging.warning(f"Internet connectivity check: Failed to connect to {host}:{port}. Error: {e}")
            return False

    @staticmethod
    async def check_ollama_service(timeout: int = 10) -> bool:
        """
        Checks if the Ollama service is running and accessible by hitting its '/api/tags' endpoint.
        This is crucial for both Ollama embeddings and Ollama LLM.
        """
        try:
            # Use aiohttp for asynchronous HTTP requests
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    return response.status == 200 # A 200 OK status indicates the service is up
        except Exception as e:
            # Catching broad exception to cover connection errors, DNS issues, etc.
            logging.debug(f"Ollama service check failed: {e}")
            return False

    @staticmethod
    async def check_embedding_model(model: str, timeout: int = 10) -> bool:
        """
        Checks if a specific Ollama embedding model is available and functional
        by performing a dummy embedding query.
        Uses `asyncio.to_thread` to run the blocking `embed_query` in a separate thread.
        """
        try:
            embeddings = OllamaEmbeddings(model=model)
            # `embed_query` can be blocking, so run it in a thread pool.
            await asyncio.wait_for(
                asyncio.to_thread(embeddings.embed_query, "test health check"),
                timeout=timeout
            )
            return True
        except Exception as e:
            logging.debug(f"Embedding model check failed for {model}: {e}")
            return False

    @staticmethod
    async def check_llm_model(model_name: str, api_key: str, timeout: int = 10) -> bool:
        """
        Checks if a Gemini LLM model is accessible and functional with a given API key.
        This involves configuring the API key and making a simple content generation request.
        Handles common Gemini API errors like invalid keys or permission issues.
        """
        # Skip check if API key is clearly a placeholder or invalid
        if not api_key or api_key.startswith(
                ("YOUR_GEMINI_API_KEY_HERE", "#AIzaSy", "AIzaSyC8finEK8KkcR0OfEVfqSZJQH48RvTEwFA")):
            logging.warning("Gemini API key is invalid or a placeholder. Skipping Gemini health check for this key.")
            return False

        async def _check_gemini():
            """Internal async function to perform the actual Gemini API call."""
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model_name)
            # `generate_content` can be blocking, so run it in a thread.
            response = await asyncio.to_thread(model_instance.generate_content, "Hello", stream=False)
            return response.text is not None # Check if a response was successfully generated

        try:
            return await asyncio.wait_for(_check_gemini(), timeout=timeout)
        except asyncio.TimeoutError:
            logging.error(f"Gemini health check for {model_name} with specific key timed out after {timeout}s.")
            return False
        except Exception as e:
            # Specific error message checks to identify common Gemini API issues
            if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e) or "API_KEY_INVALID" in str(e):
                logging.error(f"Gemini health check failed for {model_name} (key error): {e}")
            else:
                logging.error(f"Gemini health check failed for {model_name} (other error): {e}")
            return False

    @staticmethod
    def check_database_directory(db_path: str) -> bool:
        """
        Checks if the specified directory for ChromaDB is writable and accessible.
        Attempts to create the directory and write a test file.
        """
        try:
            db_dir = Path(db_path)
            db_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            test_file = db_dir / "test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test") # Attempt to write
            os.remove(test_file) # Clean up test file
            return True
        except Exception as e:
            logging.debug(f"Database directory check failed for {db_path}: {e}")
            return False

class AsyncWebScraper:
    """
    Asynchronous web scraper leveraging Selenium for dynamic content rendering
    and `ThreadPoolExecutor` to manage blocking Selenium calls concurrently.
    """

    def __init__(self, config: Config):
        self.config = config
        # Semaphore to limit the number of concurrent Selenium WebDriver instances,
        # preventing system overload.
        self.semaphore = asyncio.Semaphore(config.max_concurrent_scrapes)
        # ThreadPoolExecutor is used because Selenium WebDriver operations are blocking
        # and cannot be directly `await`ed. `asyncio.to_thread` (which uses an executor)
        # is ideal for offloading these blocking calls from the event loop.
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_scrapes)
        # List of CSS selectors to prioritize when extracting main content from a page.
        # This helps in focusing on relevant text and avoiding boilerplate/navigation.
        self.content_selectors = [
            ".markdown-section", ".markdown-body", "article", "main", ".content", "body"
        ]

    async def __aenter__(self):
        """Async context manager entry point."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point. Ensures the thread pool executor is properly
        shut down when the scraper is no longer needed, preventing resource leaks.
        `wait=True` ensures all submitted tasks complete before shutdown.
        """
        self.executor.shutdown(wait=True)

    def _create_driver(self) -> webdriver.Chrome:
        """
        Initializes and configures a headless Chrome WebDriver instance.
        Headless mode is crucial for server-side scraping without a GUI.
        Includes options to improve stability and performance in containerized environments.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
        chrome_options.add_argument("--no-sandbox")  # Required for running as root in Docker/CI environments
        chrome_options.add_argument("--disable-dev-shm-usage")  # Overcomes limited /dev/shm size in some environments
        chrome_options.add_argument("--disable-gpu")  # Disables GPU hardware acceleration, useful in headless environments
        chrome_options.add_argument("--window-size=1920,1080")  # Sets a standard window size for consistent rendering
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 RAG-Bot/1.1")
        # Suppress WebDriverManager logs for cleaner output
        os.environ['WDM_LOG_LEVEL'] = '0'
        os.environ['WDM_LOG'] = 'false'
        try:
            # ChromeDriverManager automatically downloads and manages the ChromeDriver executable.
            # This simplifies setup and ensures compatibility with the installed Chrome version.
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(self.config.page_load_timeout) # Set a timeout for full page load
            return driver
        except Exception as e:
            logging.error(
                f"Failed to create Chrome WebDriver: {e}. Ensure ChromeDriver is installable or specify path.")
            raise ScrapingError(f"WebDriver creation failed: {e}") # Propagate as a custom error

    async def _scrape_single_url(self, url: str) -> Optional[str]:
        """
        Asynchronous wrapper for the synchronous Selenium scraping logic.
        Uses a semaphore to control concurrency and `run_in_executor` to offload
        the blocking `_scrape_with_selenium` call to the thread pool.
        """
        async with self.semaphore: # Acquire a semaphore slot before starting
            try:
                # `asyncio.get_event_loop().run_in_executor` schedules a synchronous function
                # to run in the provided `executor` (ThreadPoolExecutor in this case),
                # allowing the main event loop to remain unblocked.
                content = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._scrape_with_selenium, url
                )
                return content
            except ScrapingError:
                raise # Re-raise if it's already a ScrapingError
            except Exception as e:
                logging.error(f"Async wrapper for scrape_single_url {url} failed: {str(e)}")
                return None

    def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """
        Synchronous method containing the core Selenium logic for scraping a single URL.
        It navigates to the URL, waits for the page to load, and attempts to extract
        content using a hierarchy of CSS selectors, falling back to the entire body text.
        """
        driver = None
        try:
            driver = self._create_driver() # Initialize WebDriver for this scrape
            logging.info(f"Scraping: {url}")
            driver.get(url) # Navigate to the URL

            # Wait until the document is fully loaded. This is a common way to ensure
            # JavaScript has executed and the DOM is stable.
            WebDriverWait(driver, self.config.page_load_timeout / 2).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            # Small static sleep to allow any remaining dynamic content to render,
            # though `readyState` should ideally cover most cases.
            time.sleep(self.config.scrape_timeout / 15)

            scraped_text_content = ""
            content_found = False
            # Iterate through defined content selectors to find the most relevant text
            for selector in self.content_selectors:
                try:
                    elements = []
                    # Determine how to find elements based on selector type
                    if selector.startswith('.'):
                        elements = driver.find_elements(By.CLASS_NAME, selector[1:])
                    elif selector.startswith('#'):
                        elements = driver.find_elements(By.ID, selector[1:])
                    else:
                        elements = driver.find_elements(By.TAG_NAME, selector)

                    if elements:
                        current_selector_text = ""
                        for element in elements:
                            text = element.text # Get the visible text content of the element
                            if text:
                                current_selector_text += text.strip() + "\n\n"

                        # Check if the extracted content is substantial enough
                        if len(current_selector_text.strip()) > self.config.min_content_length:
                            scraped_text_content = current_selector_text.strip()
                            content_found = True
                            logging.info(
                                f"Content found using selector '{selector}' for url: {url} (Length: {len(scraped_text_content)})")
                            break # Found good content, no need to try other selectors
                except Exception as e:
                    logging.debug(f"Selector {selector} for {url} encountered an issue: {e}")
                    continue # Try the next selector

            if not content_found:
                logging.warning(
                    f"No substantial content found with any primary selector for {url}. Attempting fallback to full body text.")
                try:
                    # Fallback: scrape the entire body text if specific selectors failed
                    body_text = driver.find_element(By.TAG_NAME, "body").text
                    if body_text and len(body_text.strip()) > self.config.min_content_length:
                        logging.info(f"Using fallback body text for {url} (Length: {len(body_text.strip())})")
                        return body_text.strip()
                    else:
                        logging.warning(f"Fallback body text for {url} also too short or empty.")
                        return None
                except Exception as e:
                    logging.error(f"Could not get body text for {url}: {e}")
                    return None
            return scraped_text_content

        except TimeoutException:
            logging.error(f"Timeout while loading or processing {url}")
            return None
        except WebDriverException as e:
            # Catches common Selenium-specific errors like "page not found", "connection refused"
            logging.error(f"WebDriver error for {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error scraping {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit() # Always quit the driver to free up resources

    async def scrape_urls(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """
        Orchestrates the asynchronous scraping of a list of URLs.
        Uses `asyncio.gather` to run multiple `_scrape_single_url` tasks concurrently.
        """
        if not urls:
            return {}
        logging.info(f"Starting to scrape {len(urls)} URLs concurrently")
        # Create a list of coroutine tasks for each URL
        tasks = [self._scrape_single_url(url) for url in urls]
        # `asyncio.gather` runs tasks concurrently. `return_exceptions=True`
        # prevents early termination if one task fails, allowing others to complete.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_content = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                # Log exceptions for individual scraping tasks
                logging.error(f"Scraping task for {url} resulted in exception: {result}")
                scraped_content[url] = None
            else:
                scraped_content[url] = result

        successful_scrapes = sum(1 for content in scraped_content.values() if content is not None)
        logging.info(f"Scraping complete. Successfully scraped {successful_scrapes}/{len(urls)} URLs.")
        return scraped_content

class EnhancedRAGSystem:
    """
    Main class for the RAG system, orchestrating data ingestion (scraping, chunking, embedding),
    retrieval (similarity, MMR), and response generation using an LLM.
    Includes robust initialization, health checks, and state persistence.
    """

    def __init__(self, config: Config):
        self.config = config
        self.setup_logging() # Configure logging early
        self.processed_urls: Set[str] = set() # Set to store URLs that have already been processed
        self.vectorstore: Optional[Chroma] = None
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.llm = None
        self.use_gemini: bool = False # Flag to track which LLM is currently in use

        # Register signal handlers for graceful shutdown on Ctrl+C or kill signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Handles OS signals (e.g., SIGINT, SIGTERM) to ensure graceful shutdown.
        This includes saving the state (processed URLs) before exiting.
        """
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.save_processed_urls() # Persist state before exiting
        sys.exit(0)

    def setup_logging(self):
        """
        Configures the Python logging system.
        Logs are directed to both a file and the console.
        Sets specific log levels for noisy external libraries to keep logs clean.
        """
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
        logging.basicConfig(
            level=logging.INFO, # Default logging level
            format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file), # Log to a file
                logging.StreamHandler(sys.stdout) # Log to console (stdout)
            ]
        )
        # Suppress verbose logging from specific libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("selenium").setLevel(logging.WARNING)
        logging.getLogger("webdriver_manager").setLevel(logging.WARNING)

    async def run_health_checks(self) -> bool:
        """
        Executes a series of health checks to verify system readiness.
        Checks include internet connectivity, Ollama service, embedding model,
        LLM availability (Gemini and Ollama fallback), and database directory access.
        Raises `HealthCheckError` for critical failures that prevent system operation.
        """
        logging.info("Running health checks...")
        overall_health_ok = True

        # Check internet connection first, as it affects other checks (Gemini, scraping)
        self.config.has_internet_access = await HealthChecker.check_internet_connection(
            host=self.config.internet_check_host,
            port=self.config.internet_check_port,
            timeout=self.config.internet_check_timeout
        )
        if self.config.has_internet_access:
            logging.info("Internet connection detected.")
        else:
            logging.warning(
                "No internet connection detected. Online features (Gemini, web scraping) will be disabled or use fallbacks.")
            overall_health_ok = False # Internet is often critical, but system can still run offline for existing data.

        # Check database directory writability
        db_dir_ok = HealthChecker.check_database_directory(self.config.db_directory)
        if not db_dir_ok:
            logging.critical(f"Database directory {self.config.db_directory} is not accessible. Halting.")
            raise HealthCheckError("Database directory check failed.") # Critical failure, cannot proceed

        # Check Ollama service and embedding model
        ollama_service_ok = await HealthChecker.check_ollama_service(self.config.health_check_timeout)
        embedding_model_ok = False
        if ollama_service_ok:
            embedding_model_ok = await HealthChecker.check_embedding_model(
                self.config.embedding_model, self.config.health_check_timeout)
        else:
            logging.warning("Ollama service not responsive. Embedding and Ollama LLM will not be available.")
            overall_health_ok = False

        if not ollama_service_ok or not embedding_model_ok:
            logging.error(
                "Core components (Ollama Service or Embedding Model) failed health check. System might not function correctly.")
            # Do not halt immediately, as the system might still be able to query existing data if LLMs are OK.
            # However, new data processing will fail.

        # Check Gemini LLM with all provided keys (if internet is available)
        gemini_llm_any_key_ok = False
        if self.config.has_internet_access and self.config.gemini_api_keys:
            for i, api_key in enumerate(self.config.gemini_api_keys):
                logging.debug(f"Pre-checking Gemini LLM ({self.config.llm_model}) with API key {i + 1}/{len(self.config.gemini_api_keys)}...")
                key_ok = await HealthChecker.check_llm_model(
                    self.config.llm_model, api_key, self.config.health_check_timeout)
                if key_ok:
                    logging.info(f"Gemini LLM pre-check successful with key {i + 1}.")
                    gemini_llm_any_key_ok = True
                    break # Found a working key, no need to check others
                else:
                    logging.warning(f"Gemini LLM pre-check failed with key {i + 1}.")
        elif not self.config.has_internet_access:
            logging.info("Skipping Gemini LLM health check due to no internet connection.")
        else:
            logging.info("No Gemini API keys configured. Skipping Gemini LLM health check.")

        # Log a summary of all health checks
        logging.info(f"Health Check Summary: Internet: {'OK' if self.config.has_internet_access else 'FAIL'}, "
                     f"DB_Dir: {'OK' if db_dir_ok else 'FAIL'}, "
                     f"OllamaSvc: {'OK' if ollama_service_ok else 'FAIL'}, "
                     f"EmbedModel: {'OK' if embedding_model_ok else 'FAIL'}, "
                     f"GeminiLLM: {'OK' if gemini_llm_any_key_ok else ('FAIL' if self.config.has_internet_access and self.config.gemini_api_keys else 'SKIPPED')}")

        if not overall_health_ok:
            logging.warning("Some critical health checks failed. Functionality may be limited or impaired.")

        logging.info("Health checks completed.")
        return overall_health_ok

    def load_processed_urls(self):
        """
        Loads the set of previously processed URLs from a JSON file.
        This prevents re-scraping and re-processing content that has already been ingested.
        """
        try:
            if Path(self.config.processed_urls_file).exists():
                with open(self.config.processed_urls_file, 'r') as f:
                    self.processed_urls = set(json.load(f))
                logging.info(f"Loaded {len(self.processed_urls)} processed URLs from {self.config.processed_urls_file}")
        except Exception as e:
            logging.error(f"Error loading processed URLs: {e}. Starting with no processed URLs.")
            self.processed_urls = set() # Reset to empty set on error

    def save_processed_urls(self):
        """
        Saves the current set of processed URLs to a JSON file.
        Ensures persistence of state for incremental data ingestion.
        """
        try:
            Path(self.config.processed_urls_file).parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(self.config.processed_urls_file, 'w') as f:
                json.dump(list(self.processed_urls), f, indent=2) # Save as a list for JSON serialization
            logging.info(f"Saved {len(self.processed_urls)} processed URLs to {self.config.processed_urls_file}")
        except Exception as e:
            logging.error(f"Error saving processed URLs: {e}")

    def load_urls_from_file(self) -> List[str]:
        """
        Reads URLs from the specified URL file.
        Filters out comments and ensures URLs start with http/https.
        """
        urls = []
        if not Path(self.config.url_file).exists():
            logging.warning(f"URL file not found: {self.config.url_file}. No URLs will be loaded.")
            return []
        try:
            with open(self.config.url_file, 'r') as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith('#') and url.startswith(('http://', 'https://')):
                        urls.append(url)
            logging.info(f"Loaded {len(urls)} URLs from {self.config.url_file}")
            return urls
        except Exception as e:
            logging.error(f"Error loading URLs from file: {e}")
            return []

    async def _initialize_gemini_llm(self, api_key: str):
        """
        Helper method to initialize and test a Gemini LLM instance with a given API key.
        Performs a test content generation to ensure the key is valid and the model is responsive.
        """
        genai.configure(api_key=api_key) # Configure the API key for Gemini
        llm_candidate = genai.GenerativeModel(self.config.llm_model)
        # Perform a small test generation to ensure connectivity and authentication
        response = await asyncio.to_thread(llm_candidate.generate_content, "Hello from Gemini init", stream=False)
        if not response.text:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                raise LLMError(f"Gemini response blocked during init test: {response.prompt_feedback.block_reason}")
            raise LLMError("Empty response from Gemini during initialization test.")
        self.llm = llm_candidate
        self.use_gemini = True # Set flag to indicate Gemini is active
        logging.info("Gemini LLM initialized and tested successfully using one of the provided API keys.")

    async def initialize_components(self):
        """
        Initializes the core RAG components: embeddings, vector store (ChromaDB), and LLM.
        Prioritizes Gemini if API keys are available and internet is connected,
        otherwise falls back to an Ollama-based LLM.
        Raises critical errors if essential components cannot be initialized.
        """
        try:
            logging.info(f"Initializing Ollama embeddings model: {self.config.embedding_model}")
            self.embeddings = OllamaEmbeddings(model=self.config.embedding_model)
            # Perform a test embedding to ensure the model loads and functions
            await asyncio.to_thread(self.embeddings.embed_query, "test embedding init")
            logging.info("Ollama Embeddings initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed to initialize Ollama embeddings: {e}. This is critical.")
            raise EmbeddingError(f"Failed to initialize Ollama embeddings: {e}")

        try:
            logging.info(f"Initializing ChromaDB vector store from: {self.config.db_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.config.db_directory,
                embedding_function=self.embeddings # ChromaDB requires an embedding function
            )
            logging.info("ChromaDB vector store initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed to initialize ChromaDB: {e}. This is critical.")
            raise DatabaseError(f"Failed to initialize ChromaDB vector store: {e}")

        gemini_initialized = False
        # Attempt to initialize Gemini LLM if internet is available and keys are provided
        if self.config.has_internet_access and self.config.gemini_api_keys:
            logging.info(
                f"Attempting to initialize Gemini LLM ({self.config.llm_model}) using {len(self.config.gemini_api_keys)} configured key(s)...")
            for i, api_key in enumerate(self.config.gemini_api_keys):
                try:
                    logging.info(f"Trying Gemini API key {i + 1}/{len(self.config.gemini_api_keys)}...")
                    # Use asyncio.wait_for to enforce a timeout on Gemini initialization
                    await asyncio.wait_for(
                        self._initialize_gemini_llm(api_key),
                        timeout=self.config.health_check_timeout
                    )
                    gemini_initialized = True
                    logging.info(f"Successfully initialized Gemini LLM with key {i + 1}.")
                    break # Success, stop trying other keys
                except asyncio.TimeoutError:
                    logging.warning(
                        f"Gemini LLM initialization with key {i + 1} timed out after {self.config.health_check_timeout}s.")
                except LLMError as e:
                    logging.warning(f"Gemini LLM initialization with key {i + 1} failed: {e}")
                except Exception as e:
                    logging.warning(
                        f"An unexpected error occurred during Gemini LLM initialization with key {i + 1}: {e}")

            if not gemini_initialized:
                logging.warning("All provided Gemini API keys failed or timed out. Falling back to Ollama.")
        elif not self.config.has_internet_access:
            logging.info("Skipping Gemini LLM initialization due to no internet connection.")
        else:
            logging.info("No valid Gemini API keys configured. Skipping Gemini LLM initialization.")

        # Fallback to Ollama LLM if Gemini could not be initialized
        if not gemini_initialized:
            try:
                logging.info(f"Initializing Ollama LLM ({self.config.fallback_llm_model}) as fallback.")
                self.llm = Ollama(model=self.config.fallback_llm_model)
                # Perform a test invocation to ensure the Ollama model is loaded and responsive
                await asyncio.to_thread(self.llm.invoke, "Hello from Ollama init")
                self.use_gemini = False
                logging.info("Ollama LLM (fallback) initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Ollama LLM (fallback): {e}.")
                # If both primary and fallback LLMs fail, it's a critical error for RAG.
                raise LLMError(f"Critical: All LLMs (Gemini and Ollama fallback) failed to initialize. Last error: {e}")

        if self.llm is None:
            # Final check to ensure at least one LLM is ready.
            raise LLMError("No LLM could be initialized (Gemini or Ollama). Cannot proceed.")

    async def process_documents(self, scraped_content: Dict[str, Optional[str]]):
        """
        Processes scraped content by chunking it, creating LangChain Documents,
        and adding them to the ChromaDB vector store.
        Handles filtering out already processed URLs.
        """
        if not self.vectorstore:
            raise DatabaseError("Vectorstore not initialized. Cannot process documents.")

        new_documents: List[Document] = []
        for url, content in scraped_content.items():
            if content and url not in self.processed_urls:
                # Use a content hash to detect potential content changes even if URL is the same.
                # Currently used in ID generation, but could be expanded for update logic.
                content_hash = hashlib.md5(content.encode('utf-8', 'replace')).hexdigest()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "timestamp": datetime.now().isoformat(),
                        "content_hash": content_hash # Store content hash in metadata
                    }
                )
                new_documents.append(doc)

        if not new_documents:
            logging.info("No new documents to process.")
            return

        logging.info(f"Processing {len(new_documents)} new documents.")
        # RecursiveCharacterTextSplitter is robust for various text types,
        # splitting by hierarchical separators and falling back to character-level.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True, # Useful for debugging and traceability
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""], # Order matters for splitting
            length_function=len # Use character length for splitting
        )
        chunks = text_splitter.split_documents(new_documents)
        logging.info(f"Split {len(new_documents)} documents into {len(chunks)} chunks.")

        if not chunks:
            logging.info("No chunks generated from new documents.")
            return

        # Add documents to ChromaDB in batches to improve performance and reduce overhead.
        # This is more efficient than adding one by one.
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                logging.info(
                    f"Adding batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} to vector store ({len(batch)} chunks)...")
                # Generate unique IDs for each chunk based on source, content hash, and start index.
                # This helps in preventing duplicate entries and allows for potential future updates.
                ids = [
                    f"{chunk.metadata['source']}_{chunk.metadata['content_hash']}_{chunk.metadata.get('start_index', idx)}"
                    for idx, chunk in enumerate(batch)]
                await asyncio.to_thread(self.vectorstore.add_documents, batch, ids=ids) # Run blocking add_documents in a thread
            except Exception as e:
                logging.error(f"Error adding document batch to vector store: {e}")
                # Consider adding retry logic here if it's a transient database error.

        logging.info(f"Persisting vector store changes...")
        await asyncio.to_thread(self.vectorstore.persist) # Persist changes to disk

        # Mark all successfully processed URLs as done.
        for doc in new_documents:
            self.processed_urls.add(doc.metadata["source"])
        self.save_processed_urls() # Save the updated set of processed URLs
        logging.info(f"Successfully processed and stored {len(new_documents)} new documents.")

    def _get_doc_hash(self, doc_content: str) -> str:
        """
        Generates a quick hash of document content (first 500 chars) for deduplication purposes.
        This is a pragmatic approach to identify similar chunks without hashing the full content,
        which might be too slow for many comparisons.
        """
        return hashlib.md5(doc_content[:500].encode('utf-8', 'replace')).hexdigest()

    def _rank_and_deduplicate_documents(self, similarity_docs: List[Document], mmr_docs: List[Document]) -> List[Dict]:
        """
        Combines and re-ranks documents retrieved by similarity search and MMR.
        This function aims to provide a diverse yet relevant set of top documents
        for the LLM context by:
        1. Deduplicating based on content hash (partial hash for efficiency).
        2. Assigning scores that prioritize similarity results but boost MMR results if they overlap.
        3. Sorting by the combined score.
        4. Applying a dynamic score drop-off filter to remove less relevant chunks.
        """
        candidates = []
        seen_content_hashes: Set[str] = set()

        # Process similarity results: assign higher initial scores
        for i, doc in enumerate(similarity_docs):
            content_hash = self._get_doc_hash(doc.page_content)
            if content_hash not in seen_content_hashes:
                # Score based on rank, higher rank (lower index) gets higher score
                score = (self.config.similarity_k - i) * 1.0
                candidates.append({'doc': doc, 'score': score, 'source_type': 'similarity'})
                seen_content_hashes.add(content_hash)

        # Process MMR results: boost score if already seen, add if new
        for i, doc in enumerate(mmr_docs):
            content_hash = self._get_doc_hash(doc.page_content)
            found_in_sim = False
            for cand in candidates:
                if self._get_doc_hash(cand['doc'].page_content) == content_hash:
                    # If an MMR doc is also a similarity doc, boost its score
                    cand['score'] += (self.config.mmr_k - i) * 0.5 # A smaller boost than initial sim score
                    cand['source_type'] += '+mmr'
                    found_in_sim = True
                    break
            if not found_in_sim and content_hash not in seen_content_hashes:
                # If MMR doc is unique, add it with a moderate score
                score = (self.config.mmr_k - i) * 0.75 # Lower initial score than pure similarity
                candidates.append({'doc': doc, 'score': score, 'source_type': 'mmr_only'})
                seen_content_hashes.add(content_hash)

        candidates.sort(key=lambda x: x['score'], reverse=True) # Sort all candidates by combined score

        # Dynamic filtering: Remove chunks if their score drops too steeply relative to the previous one.
        # This prevents including very low-relevance chunks after a few high-relevance ones.
        if len(candidates) > 1:
            filtered_candidates = []
            for i in range(len(candidates)):
                if i == 0:
                    filtered_candidates.append(candidates[i])
                else:
                    prev_score = candidates[i - 1]['score']
                    current_score = candidates[i]['score']
                    # Criteria: Large absolute drop AND large relative drop
                    if (prev_score - current_score) > 3.0 and current_score < (prev_score * 0.5):
                        logging.info(
                            f"Stopping chunk inclusion at position {i} due to significant score drop "
                            f"(from {prev_score:.2f} to {current_score:.2f})."
                        )
                        break
                    filtered_candidates.append(candidates[i])

            if len(filtered_candidates) < len(candidates):
                logging.info(
                    f"Filtered out {len(candidates) - len(filtered_candidates)} chunks "
                    f"due to consecutive score drop criteria."
                )
            candidates = filtered_candidates

        logging.info(
            f"Ranked and deduplicated {len(similarity_docs) + len(mmr_docs)} initial docs to {len(candidates)} candidates.")
        return candidates

    async def retrieve_context(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Retrieves relevant document chunks from the vector store for a given query.
        Combines similarity search and MMR for comprehensive retrieval.
        Applies re-ranking, deduplication, and a score threshold to select the best chunks.
        """
        if not self.vectorstore:
            raise RAGSystemError("Vectorstore not initialized. Cannot retrieve context.")
        logging.info(f"Retreving context for query: \"{query[:100]}...\"")
        try:
            # Perform similarity search and MMR search concurrently
            similarity_docs_task = asyncio.to_thread(
                self.vectorstore.similarity_search, query, k=self.config.similarity_k
            )
            mmr_docs_task = asyncio.to_thread(
                self.vectorstore.max_marginal_relevance_search,
                query, k=self.config.mmr_k, fetch_k=self.config.mmr_fetch_k, lambda_mult=self.config.mmr_lambda
            )
            similarity_docs, mmr_docs = await asyncio.gather(similarity_docs_task, mmr_docs_task)
            logging.info(f"Retrieved {len(similarity_docs)} similarity docs and {len(mmr_docs)} MMR docs.")

            # Rank and deduplicate the retrieved documents
            ranked_candidates = self._rank_and_deduplicate_documents(similarity_docs, mmr_docs)
            top_docs_to_consider = ranked_candidates[:self.config.final_context_chunks] # Select top N chunks

            context_parts = []
            doc_metadata_for_response = []
            actual_chunks_used = 0

            if self.config.context_score_threshold is not None:
                logging.debug(f"Applying context score threshold: {self.config.context_score_threshold}")

            # Filter chunks based on `context_score_threshold`
            for i, doc_info_item in enumerate(top_docs_to_consider):
                if self.config.context_score_threshold is None or \
                        doc_info_item['score'] >= self.config.context_score_threshold:
                    doc = doc_info_item['doc']
                    # Format context for LLM, including source and score for transparency
                    context_parts.append(
                        f"Content Block {actual_chunks_used + 1} (Original Rank: {i + 1}, Source: {doc.metadata.get('source', 'N/A')}, Score: {doc_info_item['score']:.2f}):\n{doc.page_content}")
                    doc_metadata_for_response.append({
                        'source': doc.metadata.get('source', 'Unknown'),
                        'score': round(doc_info_item['score'], 2),
                        'preview': doc.page_content[:150].replace('\n', ' ') + "...",
                        'original_rank': i + 1 # Keep track of original rank for output display
                    })
                    actual_chunks_used += 1
                elif self.config.context_score_threshold is not None:
                    logging.info(
                        f"Skipping chunk from source {doc_info_item['doc'].metadata.get('source', 'N/A')} due to low score ({doc_info_item['score']:.2f} < {self.config.context_score_threshold}).")

            if not context_parts:
                filter_msg = ""
                if self.config.context_score_threshold is not None:
                    filter_msg = f" (or all top documents had scores < {self.config.context_score_threshold})"
                logging.warning(f"No relevant documents found for query: \"{query[:100]}...\"{filter_msg}")
                return "No relevant context found.", []

            full_context_str = "\n\n---\n\n".join(context_parts)
            logging.info(
                f"Final context built from {actual_chunks_used} chunks. Total length: {len(full_context_str)} chars.")
            return full_context_str, doc_metadata_for_response
        except Exception as e:
            logging.exception(f"Error during context retrieval: {e}")
            raise RAGSystemError(f"Failed to retrieve context: {e}")

    async def generate_response(self, query: str, context: str) -> str:
        """
        Generates an LLM response to a query using the provided context.
        Includes a carefully crafted prompt to ensure the LLM sticks strictly to the context
        and avoids "hallucinations" or external knowledge.
        Handles LLM-specific errors, especially Gemini's safety filters.
        """
        if self.llm is None:
            logging.error("No LLM available for response generation.")
            raise LLMError("LLM not initialized, cannot generate response.")

        # Prompt engineering is crucial for RAG performance.
        # The prompt guides the LLM to strictly adhere to the provided context.
        # Key directives: "solely on the information presented", "ONLY the information",
        # "Do NOT use any external knowledge", "NOT mention phrases like 'the provided context'".
        prompt = f"""You are an expert technical assistant. Your task is to answer the QUESTION based *solely* on the information presented in the "Content Block(s)" below.

GUIDELINES:
Answer the QUESTION directly and comprehensively.
Use ONLY the information from the "Content Block(s)". Do NOT use any external knowledge or assumptions.
If the "Content Block(s)" do not contain the information needed to answer the QUESTION, state clearly that the information required to answer is not available.
Crucially: Throughout your entire response, including any explanation for not being able to answer, you must NOT mention phrases like "the provided context," "the documents," "the content blocks," or any similar reference to the source of your information. Your response should sound as if you are an expert providing information directly from your own knowledge base (which, for this task, is strictly limited to the "Content Block(s)").
When answering, you can incorporate information from the "Content Block(s)" by quoting or paraphrasing. However, do this naturally as part of your answer, without explicitly referencing them as external sources (e.g., instead of "The document states that...", just present the information as a fact).
Synthesize information if it spans multiple blocks to provide a coherent and complete answer.

QUESTION: {query}

Content Block(s):
{context}

ANSWER:"""
        logging.info(f"Generating response using {'Gemini' if self.use_gemini else 'Ollama'} LLM.")
        try:
            if self.use_gemini:
                # For Gemini, use `generate_content` and handle potential safety blocks.
                response = await asyncio.to_thread(self.llm.generate_content, prompt, stream=False)
                if not response.text:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                        block_reason_msg = f"Gemini response blocked due to: {response.prompt_feedback.block_reason}"
                        logging.error(block_reason_msg)
                        # Log detailed safety reasons if available
                        for candidate in response.candidates:
                            if candidate.finish_reason == 'SAFETY':
                                for rating in candidate.safety_ratings:
                                    logging.error(
                                        f"Safety Rating: Category {rating.category}, Probability {rating.probability}")
                        return f"[Error: Content generation blocked by safety filters. Reason: {response.prompt_feedback.block_reason}]"
                    raise LLMError("Empty response from Gemini LLM.")
                final_response = response.text
            else:
                # For Ollama (LangChain LLM), use `invoke`.
                response = await asyncio.to_thread(self.llm.invoke, prompt)
                if not isinstance(response, str) or not response.strip():
                    raise LLMError("Empty response from Ollama LLM.")
                final_response = response

            logging.info(f"LLM Response (first 100 chars): {final_response[:100].replace(os.linesep, ' ')}...")
            return final_response.strip()
        except Exception as e:
            logging.exception(f"Error generating LLM response: {e}")
            raise LLMError(f"Failed to generate response: {e}")

    async def run(self, query: Optional[str] = None):
        """
        Main execution flow of the RAG system.
        Orchestrates health checks, data loading, scraping, document processing, and query answering.
        Includes comprehensive error handling for each stage.
        """
        start_time = time.time()
        try:
            await self.run_health_checks() # Perform system health checks

            self.load_processed_urls() # Load URLs already processed
            await self.initialize_components() # Initialize embeddings, vector store, and LLM

            all_urls = self.load_urls_from_file() # Load all URLs from the file
            new_urls_to_scrape = [url for url in all_urls if url not in self.processed_urls] # Identify new URLs

            if new_urls_to_scrape:
                if self.config.has_internet_access:
                    logging.info(f"Found {len(new_urls_to_scrape)} new URLs to process.")
                    # Use AsyncWebScraper as an async context manager for proper resource cleanup
                    async with AsyncWebScraper(self.config) as scraper:
                        scraped_content = await scraper.scrape_urls(new_urls_to_scrape)
                    await self.process_documents(scraped_content) # Process scraped content
                else:
                    logging.warning(
                        f"Found {len(new_urls_to_scrape)} new URLs, but skipping scraping due to no internet connection. "
                        "Existing data in vector store will be used."
                    )
            else:
                logging.info(
                    "No new URLs to process, or URL file not found/empty. Using existing data in vector store.")

            if query:
                # If a query is provided, retrieve context and generate a response
                context, doc_info = await self.retrieve_context(query)
                print(f"\n--- Retrieved Context Information (Chunks sent to LLM: {len(doc_info)}) ---")
                if doc_info:
                    for i, info in enumerate(doc_info):
                        print(
                            f"  Chunk {i + 1} (Original Rank in Top {self.config.final_context_chunks}: {info.get('original_rank', 'N/A')}): Source: {info['source']}, Score: {info['score']:.2f}")
                        print(f"    Preview: {info['preview']}")
                else:
                    filter_msg = ""
                    if self.config.context_score_threshold is not None:
                        filter_msg = f" (or all top documents had scores < {self.config.context_score_threshold})"
                    print(f"  No context documents were retrieved{filter_msg}.")
                print(f"{'=' * 60}")
                response = await self.generate_response(query, context)
                print(f"\nQUERY: {query}")
                print(f"{'-' * 60}")
                print(f"RESPONSE:\n{response}")
                print(f"{'=' * 60}")
            else:
                # If no query, just inform that document processing is done
                logging.info("No query provided. System initialized and processed documents if any. Ready for queries.")
                print("System initialized. No query provided. Run with a query to get a response.")

        # Catch specific custom errors for clearer reporting
        except (EmbeddingError, DatabaseError, LLMError, HealthCheckError) as e:
            logging.critical(f"A critical RAG system error occurred: {e}", exc_info=True)
            print(f"CRITICAL ERROR: {e}. System cannot continue. Check logs for details.")
            raise # Re-raise to be caught by main for exit code
        except RAGSystemError as e:
            logging.error(f"RAG system error: {e}", exc_info=True)
            print(f"System Error: {e}. Check logs.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred in RAGSystem.run: {e}", exc_info=True)
            print(f"Unexpected Error: {e}. Check logs.")
            raise
        finally:
            self.save_processed_urls() # Always attempt to save processed URLs on exit or error
            elapsed_time = time.time() - start_time
            logging.info(f"RAGSystem.run execution (or attempt) finished in {elapsed_time:.2f} seconds.")


async def main_async():
    """
    Asynchronous entry point for the application.
    Initializes the RAG system, handles command-line arguments for queries,
    and manages the top-level execution flow.
    """
    config = Config()
    rag_system = EnhancedRAGSystem(config)

    # Determine query from command line arguments or user input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        logging.info(f"Received query from command line: {query}")
    else:
        user_input = input("Enter your question (or press Enter to just process documents): ").strip()
        query = user_input if user_input else None

    exit_code = 0 # Default exit code for success
    try:
        await rag_system.run(query=query)
    # Catch specific RAG system errors to set non-zero exit code
    except (EmbeddingError, DatabaseError, LLMError, HealthCheckError, RAGSystemError):
        exit_code = 1
    # Catch any other unexpected exceptions
    except Exception:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    # Main execution block when the script is run directly.
    # Uses `asyncio.run` to start the async event loop.
    exit_code = 1 # Default exit code for failure
    try:
        exit_code = asyncio.run(main_async())
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logging.info("Process interrupted by user (Ctrl+C).")
        print("\nProcess interrupted.")
        exit_code = 130 # Standard exit code for Ctrl+C
    except RAGSystemError as e:
        # Catch RAGSystemError specifically if it propagates to the top level
        # (though `main_async` should handle most cases)
        print(f"A system error occurred. Please check the log file ({Config().log_file}) for details.")
        exit_code = 1
    except Exception as e:
        # Catch any other unhandled exceptions at the very top level
        logging.critical(f"Unhandled exception at top level: {e}", exc_info=True)
        print(f"An unexpected critical error occurred: {e}. Check logs.")
        exit_code = 1
    finally:
        # Ensure final log message and exit with appropriate code
        logging.info(f"Application exiting with code {exit_code}.")
        sys.exit(exit_code) # Exit the process