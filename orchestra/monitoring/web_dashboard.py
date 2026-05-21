import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from orchestra.storage.db import get_db

logger = logging.getLogger(__name__)

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/runs':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            conn = get_db()
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()
            conn.close()
            
            data = [dict(r) for r in rows]
            self.wfile.write(json.dumps(data).encode('utf-8'))
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
                <head>
                    <title>Orchestra Fleet Dashboard</title>
                    <style>
                        body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
                        table { width: 100%; border-collapse: collapse; background: white; }
                        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
                        th { background: #eee; }
                        .status-completed { color: green; }
                        .status-failed { color: red; }
                        .status-running { color: blue; }
                    </style>
                </head>
                <body>
                    <h1>Orchestra Fleet Dashboard</h1>
                    <table id="runs">
                        <tr><th>Run ID</th><th>Mode</th><th>Status</th><th>Task</th><th>Cost</th></tr>
                    </table>
                    <script>
                        fetch('/api/runs').then(r => r.json()).then(data => {
                            const table = document.getElementById('runs');
                            data.forEach(run => {
                                const tr = document.createElement('tr');
                                tr.innerHTML = `
                                    <td>${run.run_id}</td>
                                    <td>${run.mode}</td>
                                    <td class="status-${run.status}">${run.status}</td>
                                    <td>${run.task.substring(0, 100)}</td>
                                    <td>$${(run.total_cost_usd || 0).toFixed(3)}</td>
                                `;
                                table.appendChild(tr);
                            });
                        });
                    </script>
                </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def start_web_dashboard(port=8080):
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    logger.info(f"Starting web dashboard on port {port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
