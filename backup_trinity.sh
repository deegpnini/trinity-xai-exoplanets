#!/data/data/com.termux/files/usr/bin/bash

echo "💾 BACKUP TRINITY FALCON LUNG"
echo "=============================="

DATA=$(date +%Y%m%d_%H%M)
BACKUP_DIR="/sdcard/Download/TRINITY_BACKUPS"
BACKUP_FILE="$BACKUP_DIR/trinity_backup_$DATA.tar.gz"

# Criar pasta
mkdir -p "$BACKUP_DIR"

# Criar backup
cd ~
tar -czf "$BACKUP_FILE" \
  trinity_falcon_lung/ \
  .bashrc \
  .bash_history 2>/dev/null

# Copiar também sem compactar (para acesso fácil)
mkdir -p "$BACKUP_DIR/trinity_$DATA"
cp -r trinity_falcon_lung/* "$BACKUP_DIR/trinity_$DATA/"

# Listar
echo "✅ Backup criado:"
ls -lh "$BACKUP_FILE"
echo ""
echo "📁 Pasta com arquivos soltos:"
ls -la "$BACKUP_DIR/trinity_$DATA/"
