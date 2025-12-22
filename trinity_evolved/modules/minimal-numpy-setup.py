"""
Numpy minimal para Termux - Configuração de build
"""
import os
import sys
import subprocess

# Configurar para build mínimo
os.environ['NPY_BLAS_ORDER'] = ''
os.environ['NPY_LAPACK_ORDER'] = ''
os.environ['NPY_DISABLE_SVML'] = '1'
os.environ['NPY_NO_SMP'] = '1'

print("🔧 Configurando build mínimo do Numpy...")

# Tentar instalar com flags mínimas
try:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        'numpy==1.19.3',
        '--no-binary', 'numpy',
        '--no-deps',
        '--install-option=--disable-optimization'
    ])
    print("✅ Numpy instalado com build mínimo")
except:
    print("⚠️  Falha no build mínimo, tentando alternativa...")
