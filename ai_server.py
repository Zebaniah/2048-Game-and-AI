import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from mcts_nn_agent import MCTSAgent

AGENT = MCTSAgent("model.json", simulations=50)


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/move":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            board = data.get("board")
            move = AGENT.best_move(board)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"move": move}).encode())
        else:
            self.send_error(404)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
