import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from mcts_nn_agent import MCTSAgent

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
AGENT = MCTSAgent(MODEL_PATH, simulations=50)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/ai_game.html"
        return super().do_GET()

    def do_POST(self):
        if self.path == "/move":
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length))
                board = data.get("board")
                move = AGENT.best_move(board)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"move": move}).encode())
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
