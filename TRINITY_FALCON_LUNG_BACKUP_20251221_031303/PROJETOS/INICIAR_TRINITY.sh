#!/bin/bash

# Cores do Menu
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

while true; do
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         🚀 TRINITY FALCON LUNG v5.0          ║${NC}"
    echo -e "${CYAN}║            PAINEL DE CONTROLE                ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}[1] 🔥 INICIAR SIMULAÇÃO v5 (Híbrido)${NC}"
    echo -e "${YELLOW}[2] 📂 Verificar Arquivos do Projeto${NC}"
    echo -e "${RED}[3] ❌ Sair${NC}"
    echo ""
    echo -n "👉 Digite sua opção: "
    read option

    case $option in
        1)
            cd ~/trinity_falcon_lung/v5_hybrid_pulmonary
            python hybrid_pulmonary_system.py
            echo ""
            read -p "Pressione Enter para voltar ao menu..."
            ;;
        2)
            echo ""
            echo "----------------------------------------"
            ls -R ~/trinity_falcon_lung | grep ":$" | head -5
            echo "----------------------------------------"
            read -p "Pressione Enter para voltar..."
            ;;
        3)
            echo "Desligando sistemas... Até logo, Hebron!"
            exit 0
            ;;
        *)
            echo "Opção inválida!"
            sleep 1
            ;;
    esac
done
