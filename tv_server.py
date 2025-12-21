import http.server
import socketserver
import subprocess

class TermuxHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/run?'):
            cmd = self.path[5:]  # remove '/run?'
            result = subprocess.getoutput(cmd)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f'<pre>{result}</pre>'.encode())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''
            <html>
            <body style="background:black;color:#0f0;">
            <h1>Termux TV Control</h1>
            <form action="/run">
            <input name="cmd" style="width:80%;font-size:20px">
            <button>Executar</button>
            </form>
            </body>
            </html>
            '''
            self.wfile.write(html.encode())

PORT = 8080
with socketserver.TCPServer(("", PORT), TermuxHandler) as httpd:
    print(f"🌐 Servidor em http://0.0.0.0:{PORT}")
    httpd.serve_forever()
