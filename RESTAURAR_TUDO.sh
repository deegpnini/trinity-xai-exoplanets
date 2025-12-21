#!/bin/bash
echo "🔄 RESTAURAÇÃO UNIVERSAL TRINITY FALCON LUNG"
echo "============================================"
echo ""
echo "Este script busca e restaura backups do Trinity"
echo ""
echo "Procurando backups disponíveis..."
echo ""
echo "1. Backups no Termux (~/):"
find ~/ -maxdepth 2 -name "*trinity*" -type d 2>/dev/null | grep -v ".termux" | head -10
echo ""
echo "2. Backups compactados:"
find ~/ -maxdepth 1 -name "TRINITY_*.tar.gz" -type f 2>/dev/null | head -5
echo ""
echo "🎯 PARA RESTAURAR MANUALMENTE:"
echo "   cp -r [pasta_backup]/trinity_falcon_lung ~/"
echo ""
echo "📞 Backup criado por: Hebron (@deegpnini)"
echo "   Data: $(date)"
