#!/data/data/com.termux/files/usr/bin/bash

echo "🚀 INSTALLING D7D_CORE v4.0..."
echo "================================"

# Dependências
echo "📦 Installing dependencies..."
pkg update -y
pkg upgrade -y
pkg install python -y
pkg install python-numpy -y

# Python packages
echo "🐍 Installing Python packages..."
pip install --upgrade pip
pip install plotext

# Verificar instalação
echo "🔍 Verifying installation..."
python -c "import numpy, plotext; print('✅ numpy', numpy.__version__); print('✅ plotext', plotext.__version__)"

# Criar estrutura
echo "📁 Creating D7D structure..."
mkdir -p {logs,output,data,config}

# Permissões
chmod +x d7d_main.py
chmod +x modules/*.py 2>/dev/null || true

echo ""
echo "✅ D7D_CORE v4.0 INSTALLED SUCCESSFULLY!"
echo ""
echo "🚀 TO RUN:"
echo "   cd ~/trinity_falcon_lung/v4_d7d_core"
echo "   python d7d_main.py"
echo ""
echo "📊 For quick test:"
echo "   python -c \"from modules import d7d_core; e=d7d_core.D7D_Core_Engine(); print('D7D Core Ready!')\""
echo ""
echo "🧠 D7D Quality: ACTIVATED"
