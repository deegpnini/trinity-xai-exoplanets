#!/bin/bash

echo "📱 TERMUX OPTIMIZED DEVELOPMENT SETUP"
echo "======================================"

# Configurar pip para usar mirrors mais rápidos
cat > ~/.pip/pip.conf << PIPCONF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
retries = 3

[install]
no-binary = numpy,pandas,scipy,matplotlib
PIPCONF

# Lista de pacotes Python que FUNCIONAM no Termux
TERMUX_COMPATIBLE_PACKAGES=(
    "requests"           # HTTP client
    "beautifulsoup4"     # Web scraping
    "flask"              # Micro web framework
    "django"             # Full web framework
    "fastapi"            # Modern API framework
    "sqlalchemy"         # ORM
    "pytest"             # Testing
    "pylint"             # Linting
    "black"              # Code formatting
    "ipython"            # Interactive Python
    "jupyter"            # Notebooks (sem compilação)
    "pyyaml"             # YAML parsing
    "python-dotenv"      # Environment variables
    "click"              # CLI framework
    "rich"               # Terminal formatting
    "typer"              # Modern CLI
    "loguru"             # Logging
    "httpx"              # Async HTTP
    "pydantic"           # Data validation
    "uvicorn"            # ASGI server
)

echo "📦 Instalando pacotes compatíveis com Termux..."
for package in "${TERMUX_COMPATIBLE_PACKAGES[@]}"; do
    echo "  → $package"
    pip install --no-binary :all: "$package" 2>/dev/null || \
    pip install "$package" --no-deps 2>/dev/null || \
    echo "  ⚠️  $package pode ter problemas"
done

# Instalar numpy alternativo (numpy-lite)
echo "🔢 Instalando numpy compatível..."
pip install numpy-lite 2>/dev/null || {
    # Se numpy-lite falhar, instalar numpy mínimo
    pip install numpy==1.19.3 --no-binary numpy --no-deps
}

# Configurar Jupyter para Termux
echo "📓 Configurando Jupyter para Android..."
pip install jupyter --no-binary jupyter 2>/dev/null
jupyter notebook --generate-config 2>/dev/null

cat >> ~/.jupyter/jupyter_notebook_config.py << JUPYTERCONFIG
# Configurações para Termux
c.NotebookApp.ip = 'localhost'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.notebook_dir = '/data/data/com.termux/files/home'
c.NotebookApp.allow_origin = '*'
c.NotebookApp.disable_check_xsrf = True
JUPYTERCONFIG

echo ""
echo "✅ SETUP DE DESENVOLVIMENTO OTIMIZADO CONCLUÍDO!"
echo ""
echo "📊 PACOTES INSTALADOS:"
pip list --format=columns | head -20
