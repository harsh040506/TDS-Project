import os
import asyncio
import logging
import sys
import time
import uuid
import base64
from io import BytesIO
from functools import wraps
from PIL import Image

from flask import Flask, request, jsonify, g

# Import from our updated rag_core
from rag_core import Config, EnhancedRAGSystem, RAGSystemError

# --- Basic Flask App and Configuration Setup ---
app = Flask(__name__)
config = Config()


# --- Logging Configuration ---
def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)

    class RequestIDFilter(logging.Filter):
        def filter(self, record):
            record.request_id = g.get('request_id', 'init')
            return True

    log_format = '%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RequestIDFilter) for f in handler.filters):
            handler.addFilter(RequestIDFilter())
    for logger_name in ["httpx", "httpcore", "werkzeug"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


setup_logging(config.log_file)

# --- Initialize RAG System ---
rag_system = EnhancedRAGSystem(config)
with app.app_context():
    app.logger.info("Starting RAG System initialization on application load...")
    try:
        asyncio.run(rag_system.initialize())
        app.logger.info("RAG System successfully initialized. Ready to accept requests.")
    except Exception as e:
        app.logger.critical(f"CRITICAL: RAG system initialization FAILED: {e}", exc_info=True)


# --- API Middleware ---
@app.before_request
def before_request_middleware():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()
    app.logger.info(f"Incoming Request: {request.method} {request.path} from {request.remote_addr}")


@app.after_request
def after_request_middleware(response):
    if 'start_time' in g:
        duration = time.time() - g.start_time
        app.logger.info(f"Response: {response.status_code} | Duration: {duration:.4f}s")
    return response


# --- API Endpoints ---
@app.route('/api/', methods=['POST'])
def handle_query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    # CORRECTED: Changed 'query' to 'question' to match the curl example.
    query = data.get('question')
    image_base64 = data.get('image')  # Optional base64 encoded image

    if not query or not isinstance(query, str):
        # CORRECTED: Updated the error message to reflect the new key.
        return jsonify({"error": "Missing or invalid 'question' field"}), 400

    if not rag_system.is_initialized:
        app.logger.error("Query received but RAG system is not initialized. Service unavailable.")
        return jsonify({"error": "Service is not ready, initialization may have failed."}), 503

    try:
        # Process image if provided
        image = None
        if image_base64:
            try:
                # Remove the data URL prefix if present
                if image_base64.startswith('data:image'):
                    image_base64 = image_base64.split(',')[1]

                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))
                app.logger.info("Successfully decoded and processed image")
            except Exception as e:
                app.logger.error(f"Failed to process image: {e}")
                return jsonify({"error": "Invalid image data"}), 400

        result = asyncio.run(rag_system.answer_query(query, image=image))
        return jsonify(result), 200
    except RAGSystemError as e:
        app.logger.error(f"RAG system error processing query: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing the request."}), 500
    except Exception as e:
        app.logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/health', methods=['GET'])
def health_check():
    if rag_system.is_initialized:
        return jsonify({"status": "ok", "message": "RAG system is initialized."}), 200
    else:
        return jsonify({"status": "error", "message": "RAG system is not initialized."}), 503


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)