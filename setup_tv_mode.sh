#!/data/data/com.termux/files/usr/bin/bash

echo "📺 CONFIGURANDO TERMUX PARA TV"
echo "=============================="

# Instalar dependências básicas
echo "[1/6] Instalando pacotes..."
pkg update -y
pkg upgrade -y
pkg install python nodejs openssh tigervnc -y

# Configurar interface para TV
echo "[2/6] Configurando interface..."
mkdir -p ~/.termux
cat > ~/.termux/termux.properties << 'TERMPROP'
font-size = 16
use-black-ui = true
bell-character = ignore
terminal-margin-horizontal = 20
TERMPROP
termux-reload-settings

# Configurar SSH para acesso remoto
echo "[3/6] Configurando SSH..."
passwd << 'PASS'
termuxtv123
termuxtv123
PASS
sshd

# Configurar servidor web simples
echo "[4/6] Configurando servidor web..."
cat > ~/tv_server.py << 'PYTHON_SERVER'
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
PYTHON_SERVER

# Obter informações de rede
echo "[5/6] Obtendo informações de rede..."
IP=$(ifconfig wlan0 2>/dev/null | grep 'inet ' | awk '{print $2}')
if [ -z "$IP" ]; then
    IP=$(termux-wifi-connectioninfo 2>/dev/null | grep -o '"ip":[^,]*' | cut -d'"' -f4)
fi

# Mostrar opções de acesso
echo "[6/6] Configuração completa!"
echo ""
echo "📡 OPÇÕES DE ACESSO PARA TV:"
echo ""
echo "1. 📱 ESPELHAMENTO DE TELA:"
echo "   • Abra 'Espelhamento' no celular"
echo "   • Selecione sua TV LG"
echo ""
echo "2. 🌐 TERMINAL WEB:"
echo "   • No navegador da TV, acesse:"
echo "   http://${IP:-seu-ip}:8080"
echo ""
echo "3. 🔐 SSH:"
echo "   • Use app SSH na TV"
echo "   • Host: ${IP:-seu-ip}"
echo "   • Porta: 8022"
echo "   • Usuário: $(whoami)"
echo "   • Senha: termuxtv123"
echo ""
echo "4. 🖥️ VNC:"
echo "   • Configure: vncserver -localhost"
echo "   • Porta: 5901"
echo ""
echo "🔧 COMANDOS ÚTEIS:"
echo "   • Iniciar servidor web: python ~/tv_server.py"
echo "   • Iniciar SSH: sshd"
echo "   • Ver IP: ifconfig wlan0"
echo ""
echo "✅ PRONTO PARA TV!"
