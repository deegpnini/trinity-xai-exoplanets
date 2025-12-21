#!/bin/bash

# Cores
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║      🚀 TRINITY FALCON LUNG CENTER       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo "1. 🌪️  Iniciar Sistema v5 (Híbrido Pulmonar)"
    echo "2. 🧠  Iniciar Sistema v4 (D7D Core)"
    echo "3. 📂  Verificar Arquivos de Projeto"
    echo "4. ❌  Sair"
    echo ""
    echo -n "Escolha uma opção [1-4]: "
    read option

    case $option in
        1)
            cd ~/trinity_falcon_lung/v5_hybrid_pulmonary
            python hybrid_pulmonary_system.py
            echo ""
            read -p "Pressione Enter para voltar ao menu..."
            ;;
        2)
            if [ -d "~/trinity_falcon_lung/v4_d7d_core" ]; then
                cd ~/trinity_falcon_lung/v4_d7d_core
                # python d7d_launcher.py (se existir)
                echo "Simulando v4..."
            else
                echo "v4 Core aguardando restauração completa."
            fi
            sleep 2
            ;;
        3)
            echo "----------------------------------------"
            ls -R ~/trinity_falcon_lung | grep ":$" | head -5
            echo "----------------------------------------"
            read -p "Pressione Enter para voltar..."
            ;;
        4)
            echo "Saindo... Até logo, Hebron!"
            exit 0
            ;;
        *)
            echo "Opção inválida!"
            sleep 1
            ;;
    esac
done
