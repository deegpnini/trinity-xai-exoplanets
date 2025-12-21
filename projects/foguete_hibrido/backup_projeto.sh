#!/bin/bash

echo "🚀 BACKUP DO PROJETO FOGUETE HÍBRIDO BR"
echo "========================================"

DATA=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/backups/foguete_hibrido"
BACKUP_FILE="$BACKUP_DIR/foguete_hibrido_backup_$DATA.tar.gz"

# Criar diretório de backup se não existir
mkdir -p "$BACKUP_DIR"

echo "📦 Compactando projeto..."
cd ~/projects/foguete_hibrido
tar -czf "$BACKUP_FILE" .

# Verificar tamanho
TAMANHO=$(du -h "$BACKUP_FILE" | cut -f1)

echo ""
echo "✅ BACKUP CONCLUÍDO!"
echo "📁 Arquivo: $BACKUP_FILE"
echo "📊 Tamanho: $TAMANHO"
echo ""
echo "📋 CONTEÚDO DO BACKUP:"
tar -tzf "$BACKUP_FILE" | head -20
echo "..."
echo ""
echo "🔄 Para restaurar:"
echo "   cd ~/projects/"
echo "   tar -xzf $BACKUP_FILE"
echo ""
