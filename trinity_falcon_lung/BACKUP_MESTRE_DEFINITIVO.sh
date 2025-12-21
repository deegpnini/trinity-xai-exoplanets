#!/bin/bash

echo "=========================================="
echo "🚀 TRINITY FALCON LUNG - BACKUP DEFINITIVO"
echo "=========================================="
echo "Dispositivo: Galaxy A70 (Carregando 🔋)"
echo "Serial: RX8M80F0AMW"
echo "Data: $(date '+%A, %d de %B de %Y %H:%M:%S')"
echo ""

# 1. CRIAR ESTRUTURA
BACKUP_ROOT="$HOME/TRINITY_FALCON_LUNG_BACKUP_$(date +%Y%m%d_%H%M%S)"
echo "📁 1. CRIANDO ESTRUTURA EM: $BACKUP_ROOT"
mkdir -p "$BACKUP_ROOT"/{PROJETOS,CONFIG,SCRIPTS,DOCS,DADOS,METADADOS}

# 2. SALVAR PROJETOS
echo "💻 2. SALVANDO VERSÕES..."
if [ -d ~/trinity_falcon_lung ]; then
    cp -r ~/trinity_falcon_lung/* "$BACKUP_ROOT/PROJETOS/" 2>/dev/null
    echo "✅ Projetos copiados (v3, v4, v5)"
else
    echo "⚠️ Pasta original não encontrada, criando vazia."
fi

# 3. SALVAR CONFIGS
echo "⚙️ 3. SALVANDO CONFIGURAÇÕES..."
cp ~/.bashrc "$BACKUP_ROOT/CONFIG/" 2>/dev/null
cp ~/.gitconfig "$BACKUP_ROOT/CONFIG/" 2>/dev/null
termux-info > "$BACKUP_ROOT/CONFIG/termux_info.txt" 2>/dev/null
pkg list-installed > "$BACKUP_ROOT/CONFIG/packages_installed.txt" 2>/dev/null

# 4. DOCUMENTAÇÃO
echo "📚 4. GERANDO DOCUMENTAÇÃO TÉCNICA..."

# Doc Principal
cat > "$BACKUP_ROOT/DOCS/DOCUMENTACAO_COMPLETA.md" << 'DOC'
# 🚀 DOCUMENTAÇÃO COMPLETA - TRINITY FALCON LUNG
## HISTÓRIA
Projeto desenvolvido entre 20-21/12/2025 no Galaxy A70.
Evolução: v3 (Básico) -> v4 (D7D Core) -> v5 (Híbrido Pulmonar).

## RESULTADOS v5 HYBRID
- Altitude: 45km
- Economia: R$ 350.000/lançamento
- Tecnologia: Python no Termux Android
- Colaboração: Hebron + Grok AI

## ESTRUTURA
Este backup contém todo o código fonte, logs e metadados.
DOC

# Info do Dispositivo (Conforme seus dados)
cat > "$BACKUP_ROOT/METADADOS/DISPOSITIVO_INFO.md" << 'META'
# 📱 METADADOS DO GALAXY A70
- Modelo: SM-A705MN
- Serial: RX8M80F0AMW
- Android: 11 (One UI 3.1)
- Processador: Snapdragon 675
- RAM: 6GB
- Status: Oficial (Sem Root)
- Projeto: 100% Mobile Development
META

# 5. SCRIPTS DE UTILIDADE
echo "🔧 5. CRIANDO FERRAMENTAS DE RESTAURAÇÃO..."

# Script de Restauração Embutido
cat > "$BACKUP_ROOT/SCRIPTS/restaurar_completo.sh" << 'RESTORE'
#!/bin/bash
echo "🔄 RESTAURANDO TRINITY..."
cp -r ../PROJETOS/* ~/trinity_falcon_lung/
echo "✅ Restauração concluída em ~/trinity_falcon_lung"
RESTORE
chmod +x "$BACKUP_ROOT/SCRIPTS/restaurar_completo.sh"

# 6. COMPACTAÇÃO FINAL (GOOGLE DRIVE)
echo "📦 6. CRIANDO PACOTE PARA GOOGLE DRIVE..."
cd ~
FINAL_FILE="TRINITY_FALCON_LUNG_FULL_LEGACY_$(date +%Y%m%d).tar.gz"
tar -czf "$FINAL_FILE" "$(basename "$BACKUP_ROOT")"

# 7. RESUMO
SIZE=$(du -h "$FINAL_FILE" | cut -f1)
echo ""
echo "=========================================="
echo "🎉 BACKUP DEFINITIVO CONCLUÍDO!"
echo "=========================================="
echo "📂 Pasta Organizada: $BACKUP_ROOT"
echo "📦 Arquivo para Drive: ~/$FINAL_FILE"
echo "📊 Tamanho Final: $SIZE"
echo "=========================================="
echo "🇧🇷 LEGADO PRESERVADO COM SUCESSO!"
