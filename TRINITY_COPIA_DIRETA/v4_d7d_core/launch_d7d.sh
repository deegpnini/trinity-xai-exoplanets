#!/data/data/com.termux/files/usr/bin/bash

echo "🚀 D7D_CORE LAUNCHER v4.0"
echo "========================="

cd "$(dirname "$0")"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado!"
    echo "Instale com: pkg install python"
    exit 1
fi

# Verificar se estamos no diretório certo
if [ ! -f "d7d_launcher.py" ]; then
    echo "❌ Arquivo d7d_launcher.py não encontrado!"
    echo "Certifique-se de estar em ~/trinity_falcon_lung/v4_d7d_core"
    exit 1
fi

# Instalar plotext se necessário
python3 -c "import plotext" 2>/dev/null || {
    echo "📦 Instalando plotext para gráficos..."
    pip install plotext
}

# Executar
echo "🎮 Iniciando D7D Control Panel..."
echo ""

python3 d7d_launcher.py

echo ""
echo "🏁 D7D Core finalizado."
echo "📁 Diretório: $(pwd)"
