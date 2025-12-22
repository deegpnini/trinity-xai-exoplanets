import http.server
import socketserver
import subprocess
import urllib.parse
from datetime import datetime

PORT = 9090  # Porta fácil de lembrar

class TVHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Interface principal
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = '''<!DOCTYPE html>
<html>
<head>
    <title>📱 Termux → 📺 TV LG</title>
    <meta name="viewport" content="width=1280, initial-scale=1.0">
    <style>
        * { font-size: 28px; }
        body { background: #000; color: #0f0; font-family: monospace; padding: 30px; }
        h1 { color: #0ff; text-align: center; font-size: 48px; }
        .terminal { 
            background: #111; 
            padding: 25px; 
            border-radius: 15px; 
            border: 3px solid #0f0;
            min-height: 400px;
            margin: 20px 0;
            overflow: auto;
        }
        input { 
            width: 80%; 
            padding: 20px; 
            font-size: 32px; 
            margin: 10px;
            background: #222;
            color: #0f0;
            border: 2px solid #0f0;
        }
        button { 
            padding: 20px 40px; 
            font-size: 32px; 
            margin: 10px;
            background: #00a;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
        .quick-btn { 
            display: inline-block; 
            padding: 15px; 
            margin: 5px; 
            background: #333; 
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>✅ CONEXÃO ESTABELECIDA!</h1>
    <div style="text-align:center;color:#0ff;font-size:32px;">
        📡 Mesma rede local detectada
    </div>
    
    <div class="terminal" id="output">
        $ Bem-vindo ao Termux na TV LG!<br>
        > Digite comandos abaixo...
    </div>
    
    <div style="text-align:center;">
        <div class="quick-btn" onclick="run('pwd')">📍 Local</div>
        <div class="quick-btn" onclick="run('ls -la')">📂 Listar</div>
        <div class="quick-btn" onclick="run('neofetch')">🖥️ Sistema</div>
        <div class="quick-btn" onclick="run('date')">🕐 Data</div>
        <div class="quick-btn" onclick="run('termux-battery-status')">🔋 Bateria</div>
    </div>
    
    <div style="text-align:center; margin: 20px 0;">
        <input type="text" id="cmd" placeholder="Digite comando (ex: ls, python, etc)...">
        <br>
        <button onclick="execute()">🚀 EXECUTAR</button>
        <button onclick="document.getElementById('cmd').value=''">🧹 LIMPAR</button>
    </div>
    
    <script>
        function run(cmd) {
            document.getElementById('cmd').value = cmd;
            execute();
        }
        
        function execute() {
            let cmd = document.getElementById('cmd').value;
            let output = document.getElementById('output');
            
            output.innerHTML += '<br>$ ' + cmd + '<br>';
            
            fetch('/exec?cmd=' + encodeURIComponent(cmd))
                .then(r => r.text())
                .then(text => {
                    output.innerHTML += '<pre style="color:#ccc">' + text + '</pre>';
                    output.scrollTop = output.scrollHeight;
                });
            
            document.getElementById('cmd').value = '';
            document.getElementById('cmd').focus();
        }
    </script>
</body>
</html>'''
            self.wfile.write(html.encode('utf-8'))
        
        elif self.path.startswith('/exec'):
            # Executar comando
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            cmd = params.get('cmd', [''])[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            
            if cmd:
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                    output = result.stdout
                    if result.stderr:
                        output += "\n--- ERROS ---\n" + result.stderr
                except Exception as e:
                    output = f"Erro: {str(e)}"
            else:
                output = "Comando vazio"
            
            self.wfile.write(output.encode('utf-8'))
        
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        # Silenciar logs
        pass

print("=" * 60)
print("📺 SERVIDOR TERMUX PARA TV LG")
print("=" * 60)

# Obter IP automaticamente
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
YOUR_IP = s.getsockname()[0]
s.close()

print(f"✅ IP do celular: {YOUR_IP}")
print(f"📺 IP da TV: 192.168.3.16")
print(f"🌐 Porta: {PORT}")
print("")
print("📱 NA TV LG:")
print("1. Abra o NAVEGADOR WEB")
print(f"2. Digite: http://{YOUR_IP}:{PORT}")
print("3. Use os botões ou digite comandos")
print("")
print("⚡ Servidor iniciando...")
print("=" * 60)

with socketserver.TCPServer(("", PORT), TVHandler) as httpd:
    httpd.serve_forever()
