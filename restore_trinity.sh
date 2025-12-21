#!/data/data/com.termux/files/usr/bin/bash

echo "🔄 RESTAURAÇÃO TRINITY FALCON LUNG"
echo "================================"

# Ativar storage
termux-setup-storage

# Procurar backups
BACKUP_FOUND=0
for backup in /sdcard/trinity_*.tar.gz /sdcard/*trinity* /sdcard/*falcon*; do
    if [ -f "$backup" ] || [ -d "$backup" ]; then
        echo "📦 Backup encontrado: $backup"
        BACKUP_FOUND=1
        
        if [[ "$backup" == *.tar.gz ]]; then
            echo "📤 Extraindo..."
            tar -xzf "$backup" -C ~/
        elif [ -d "$backup" ]; then
            echo "📂 Copiando pasta..."
            cp -r "$backup" ~/
        fi
        break
    fi
done

if [ $BACKUP_FOUND -eq 0 ]; then
    echo "❌ Nenhum backup encontrado!"
    echo ""
    echo "📝 INSTRUÇÕES:"
    echo "1. Abra o Termux ANTIGO (Play Store)"
    echo "2. Execute:"
    echo "   cd ~ && tar -czf /sdcard/trinity_backup_now.tar.gz comet_c2025_r3_mission/"
    echo "3. Volte aqui e rode este script novamente"
    exit 1
fi

# Instalar Python se necessário
if ! command -v python &> /dev/null; then
    echo "🐍 Instalando Python..."
    pkg install python -y
fi

# Testar
echo "🚀 Testando projeto Trinity..."
if [ -d "~/comet_c2025_r3_mission/falcon_lung_project" ]; then
    cd ~/comet_c2025_r3_mission/falcon_lung_project
    python trinity_falcon_lung_v2.py
else
    echo "📁 Procurando projeto..."
    find ~ -name "trinity_falcon_lung_v2.py" 2>/dev/null
fi

echo ""
echo "✅ Processo completo!"
