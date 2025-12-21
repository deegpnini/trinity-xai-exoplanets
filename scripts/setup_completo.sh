#!/bin/bash

echo "=========================================="
echo "🚀 SETUP COMPLETO TERMUX - GALAXY A70"
echo "=========================================="
echo ""

# 1. Atualizar sistema
echo "🔄 ATUALIZANDO SISTEMA..."
pkg update -y && pkg upgrade -y

# 2. Instalar pacotes base
echo "📦 INSTALANDO PACOTES BASE..."
pkg install -y \
    git \
    python \
    nodejs \
    nano \
    wget \
    curl \
    openssh \
    termux-api

# 3. Configurar Python
echo "🐍 CONFIGURANDO PYTHON..."
pip install --upgrade pip
pip install --no-binary :all: \
    requests \
    flask \
    fastapi \
    pytest \
    pylint \
    black \
    ipython \
    rich

# 4. Configurar Git
echo "🔧 CONFIGURANDO GIT..."
git config --global user.name "Hebron"
git config --global user.email "deegp.nini@gmail.com"

# 5. Configurar SSH
echo "🔐 CONFIGURANDO SSH..."
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -C "deegp.nini@gmail.com" -f ~/.ssh/id_ed25519 -N ""
fi

# 6. Criar estrutura
echo "📁 CRIANDO ESTRUTURA..."
mkdir -p ~/projects/{personal,work,experiments}
mkdir -p ~/scripts
mkdir -p ~/backups

# 7. Mensagem final
echo ""
echo "=========================================="
echo "🎉 SETUP COMPLETO CONCLUÍDO!"
echo "=========================================="
echo ""
echo "🔑 SUA CHAVE SSH (para GitHub):"
cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "Chave não gerada"
echo ""
echo "🚀 COMANDOS DISPONÍVEIS:"
echo "   • pipi <pacote>   - Instalar sem binary"
echo "   • update          - Atualizar pacotes"
echo "   • backup          - Backup rápido"
echo ""
echo "🇧🇷 AMBIENTE PRONTO PARA DESENVOLVIMENTO!"
