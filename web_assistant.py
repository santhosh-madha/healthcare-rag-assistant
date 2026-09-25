"""Serve the educational assistant locally: python3 web_assistant.py."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from structured_healthcare import (
    CACHE, MODEL_NAME, SentenceTransformer, STRUCTURED_PROMPT,
    build_healthcare_context, generate_answer, load_index, read_corpus,
    search, validate_response,
)

PAGE = Path(__file__).parent / "web" / "index.html"


def ask(model, index, passages, question):
    """Return validated output with evidence; never expose rejected claims."""
    results = search(model, index, passages, question, limit=3)
    raw = generate_answer(question, build_healthcare_context(results),
                          system_prompt=STRUCTURED_PROMPT, response_format="json")
    response = validate_response(raw, results)
    return {"response": response, "sources": results}


def make_handler(model, index, passages):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.send(404, b'{"error":"Not found"}')
                return
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Origin") != origin:
                self.send(403, b'{"error":"Open the app using its local URL."}')
                return
            if self.path != "/ask":
                self.send(404, b'{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10000:
                    raise ValueError("Request is too large or empty.")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected JSON.")
                data = json.loads(self.rfile.read(length))
                question = data.get("question") if isinstance(data, dict) else None
                if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                    raise ValueError("Enter a question of 1–2000 characters.")
            except (ValueError, UnicodeError) as error:
                self.send(400, json.dumps({"error": str(error)}).encode())
                return
            try:
                result = ask(model, index, passages, question.strip())
            except ValueError:
                self.send(422, json.dumps({"error": "The response failed evidence validation and was withheld. Try rephrasing your question."}).encode())
                return
            except (RuntimeError, OSError):
                self.send(503, json.dumps({"error": "Could not generate an answer. Check that Ollama is running with llama3.1:8b, then try again."}).encode())
                return
            self.send(200, json.dumps(result).encode())
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        passages, corpus_hash = read_corpus()
        print("Loading model and saved index...", flush=True)
        model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE), device="cpu")
        index = load_index(model, passages, corpus_hash)
        server = HTTPServer(("127.0.0.1", args.port), make_handler(model, index, passages))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Open http://127.0.0.1:{server.server_port} — Ctrl-C to stop.", flush=True)
    print("Questions are independent. Restart after rebuilding the index.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
