#!/data/data/com.termux/files/usr/bin/bash

echo "🔐 TRINITY FALCON LUNG - BACKUP COMPLETO"
echo "========================================="
echo ""

# Criar backup em locais múltiplos
BACKUP_NAME="TRINITY_BACKUP_$(date +%Y%m%d_%H%M)"
BACKUP_TERMUX="$HOME/$BACKUP_NAME"
BACKUP_ANDROID="/sdcard/Download/$BACKUP_NAME"

echo "📁 Criando backup no Termux..."
mkdir -p "$BACKUP_TERMUX"
cp -r ~/trinity_falcon_lung "$BACKUP_TERMUX/"
cp ~/.bashrc "$BACKUP_TERMUX/" 2>/dev/null || true

echo "📱 Criando backup na pasta Download..."
mkdir -p "$BACKUP_ANDROID"
cp -r ~/trinity_falcon_lung "$BACKUP_ANDROID/"

# Criar arquivo de instruções
cat > "$BACKUP_TERMUX/COMO_RESTAURAR.txt" << INSTRUCOES
COMO RESTAURAR O TRINITY FALCON LUNG:
======================================

1. NO TERMUX (se backup estiver em ~/):
   cp -r $BACKUP_NAME/trinity_falcon_lung ~/

2. SE BAIXOU DO GOOGLE DRIVE:
   • Coloque a pasta no /sdcard/Download/
   • No Termux: termux-setup-storage
   • Copie: cp -r /sdcard/Download/$BACKUP_NAME/trinity_falcon_lung ~/

3. TESTE:
   cd ~/trinity_falcon_lung/v5_hybrid_pulmonary
   python hybrid_pulmonary_system.py

CONTÉM:
✓ v3_basic/ - Sistema básico
✓ v4_d7d_core/ - Núcleo avançado  
✓ v5_hybrid_pulmonary/ - Sistema híbrido completo

Backup criado em: $(date)
Por: Hebron (@deegpnini)
INSTRUCOES

cp "$BACKUP_TERMUX/COMO_RESTAURAR.txt" "$BACKUP_ANDROID/"

# Compactar versão para nuvem
echo "📦 Compactando para Google Drive..."
cd ~
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

echo ""
echo "✅ BACKUP CONCLUÍDO!"
echo ""
echo "📍 LOCAIS DO BACKUP:"
echo "   1. Termux: ~/$BACKUP_NAME"
echo "   2. Android: /sdcard/Download/$BACKUP_NAME"
echo "   3. Compactado: ~/${BACKUP_NAME}.tar.gz"
echo ""
echo "📊 TAMANHOS:"
echo "   - Pasta: $(du -sh "$BACKUP_TERMUX" | cut -f1)"
echo "   - Compactado: $(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)"
echo ""
echo "☁️ PARA GOOGLE DRIVE:"
echo "   1. Abra app 'Arquivos'"
echo "   2. Navegue até: Internal storage/termux/home/"
echo "   3. Toque em: ${BACKUP_NAME}.tar.gz"
echo "   4. Toque em 'Compartilhar' → 'Salvar no Drive'"
echo ""
echo "🔒 BACKUP SALVO EM 3 LOCAIS DIFERENTES"
echo "🇧🇷 CONHECIMENTO GARANTIDO!"

# Mostrar caminhos completos
echo ""
echo "📁 CAMINHOS COMPLETOS:"
echo "   • Termux: $BACKUP_TERMUX"
echo "   • Android: $BACKUP_ANDROID"
echo "   • Compactado: ~/${BACKUP_NAME}.tar.gz"
