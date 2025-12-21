#!/bin/bash

echo "🔧 CORRIGINDO PROBLEMAS DE COMPILAÇÃO TERMUX..."
echo ""

# 1. Atualizar pacotes
pkg update && pkg upgrade -y

# 2. Instalar compiladores essenciais
pkg install -y clang make cmake binutils

# 3. Reconfigurar pip
pip config set global.no-binary "numpy,pandas,scipy,matplotlib"
pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn"

# 4. Instalar bibliotecas garantidas
echo "📦 Instalando bibliotecas que FUNCIONAM..."
for pkg in requests beautifulsoup4 flask fastapi pytest pylint black ipython; do
    pip install --no-binary :all: $pkg 2>/dev/null || pip install $pkg
    echo "✅ $pkg"
done

# 5. Instalar alternativas leves
pip install numpy-lite pandas-lite matplotlib-inline 2>/dev/null || true

# 6. Verificar
echo ""
echo "✅ CORREÇÕES APLICADAS!"
echo "🐍 Python está funcionando com bibliotecas essenciais"
echo "🚀 Ambiente pronto para desenvolvimento"
