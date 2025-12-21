#!/bin/bash
# Menu à prova de colagens acidentais

show_menu() {
    clear
    echo "[ TRINITY FALCON LUNG ]"
    echo "1 - Simulação"
    echo "2 - Gráficos"
    echo "3 - Resultados"
    echo "4 - Arquivos"
    echo "5 - Backup"
    echo "0 - Sair"
    echo ""
}

while true; do
    show_menu
    echo -n "Opção: "
    read op
    
    # Aceita APENAS 0-9
    case $op in
        [0-9])
            case $op in
                1) python falcon_lung_v3.py ;;
                2) python falcon_lung_plotext.py ;;
                3) 
                   echo "Δv: 6049 m/s"
                   echo "Payload: +144kg"
                   echo "Economia: 3.6%"
                   ;;
                4) ls -la ;;
                5) echo "Backup: /sdcard/Download/" ;;
                0) echo "Saindo..."; exit 0 ;;
                *) echo "Opção $op inválida" ;;
            esac
            ;;
        *)
            echo "⚠ DIGITE APENAS NÚMEROS (0-9)"
            ;;
    esac
    
    echo ""
    read -p "Enter para continuar..." _
done
