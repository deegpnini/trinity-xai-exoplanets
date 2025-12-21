#!/data/data/com.termux/files/usr/bin/bash

echo "🔄 Instalando Trinity Falcon Lung no Termux F-Droid..."

# Dependências
pkg install python -y
pkg install git -y

# Instalar matplotlib se quiser gráficos depois
# pkg install python-numpy -y
# pip install matplotlib

echo "✅ Instalação completa!"
echo ""
echo "🚀 PARA EXECUTAR:"
echo "cd ~/trinity_falcon_lung"
echo "python falcon_lung_v3.py"
echo ""
echo "📚 Documentação em: README_TRINITY.md"
