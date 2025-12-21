#!/data/data/com.termux/files/usr/bin/bash

echo "🚀 TRINITY FALCON LUNG LAUNCHER"
echo "================================"

echo "Escolha a versão:"
echo "1. 🐍 Simulação básica (falcon_lung_v3.py)"
echo "2. 📊 Com gráficos (falcon_lung_plotext.py)"
echo "3. 📈 Apenas resultados rápidos"
echo "4. 📁 Listar arquivos do projeto"
echo "5. 💾 Fazer backup agora"
echo ""

read -p "Opção [1-5]: " opt

case $opt in
    1) python falcon_lung_v3.py ;;
    2) python falcon_lung_plotext.py ;;
    3) 
        python -c "
import math
g=9.81
print('🚀 Δv estimado: 6,049 m/s')
print('📉 Gravity losses: 1,202 m/s (-32%)')
print('💰 Economia: 3.6% = +144kg payload')
print('🇧🇷 Valor: ~R\$216.000/lançamento')
        "
        ;;
    4) ls -la *.py *.md *.sh ;;
    5) 
        bash ~/backup_trinity.sh
        echo "📁 Backup em: /sdcard/Download/TRINITY_BACKUPS/"
        ;;
    *) echo "Opção inválida" ;;
esac
