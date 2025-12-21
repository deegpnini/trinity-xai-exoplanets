#!/bin/bash

echo "🚀 INICIANDO SETUP PROFISSIONAL..."
echo "Isso pode levar alguns minutos. Aguarde..."

# Atualizar sistema
pkg update -y && pkg upgrade -y

# Instalar pacotes base
pkg install -y git wget curl nano python nodejs clang make termux-api proot

# Configurar Python
pip install --upgrade pip virtualenv virtualenvwrapper
pip install numpy pandas matplotlib requests flask

# Configurar Git
git config --global user.name "Hebron"
git config --global user.email "deegp.nini@gmail.com"

# Gerar SSH
ssh-keygen -t ed25519 -C "deegp.nini@gmail.com" -f ~/.ssh/id_ed25519 -N ""

echo ""
echo "✅ SETUP BASE COMPLETO!"
echo ""
echo "🔑 SUA CHAVE SSH:"
cat ~/.ssh/id_ed25519.pub
echo ""
echo "📋 Copie e cole no GitHub"
echo "🇧🇷 Agora você está pronto para codificar como um profissional!"
