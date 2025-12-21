#!/data/data/com.termux/files/usr/bin/bash

while true; do
    clear
    echo "╔══════════════════════════════════╗"
    echo "║   🚀 TRINITY FALCON LUNG v3.0    ║"
    echo "╚══════════════════════════════════╝"
    echo ""
    echo "1️⃣  Simulação completa"
    echo "2️⃣  Versão com gráficos" 
    echo "3️⃣  Resultados rápidos"
    echo "4️⃣  Arquivos do projeto"
    echo "5️⃣  Fazer backup"
    echo "0️⃣  Sair"
    echo ""
    
    # Limpar input buffer
    while read -t 0; do read -n 10000 -t 0.1; done
    
    # Ler com validação
    read -p "👉 Digite apenas o NÚMERO (0-5): " opcao
    
    # Remover espaços e caracteres especiais
    opcao=$(echo "$opcao" | tr -d '[:space:]' | grep -E '^[0-5]$')
    
    if [ -z "$opcao" ]; then
        echo ""
        echo "⚠️  Digite APENAS um número de 0 a 5!"
        echo "   (Sem letras, sem colar texto)"
        sleep 2
        continue
    fi
    
    case $opcao in
        1)
            echo ""
            echo "🚀 Iniciando simulação básica..."
            echo "══════════════════════════════════"
            python falcon_lung_v3.py
            ;;
        2)
            echo ""
            echo "📊 Iniciando versão com gráficos..."
            echo "══════════════════════════════════"
            python falcon_lung_plotext.py
            ;;
        3)
            echo ""
            echo "📈 RESULTADOS VALIDADOS:"
            echo "══════════════════════════════════"
            echo "   Δv total: 6.049 m/s"
            echo "   Gravity losses: 1.202 m/s"
            echo "   Redução: 32%"
            echo "   Economia: 3.6% combustível"
            echo "   Payload extra: +144 kg"
            echo "   Valor: ~R$216.000/lançamento"
            echo ""
            ;;
        4)
            echo ""
            echo "📁 ARQUIVOS DO PROJETO:"
            echo "══════════════════════════════════"
            echo "Python scripts:"
            ls *.py
            echo ""
            echo "Documentação:"
            ls *.md 2>/dev/null || echo "(não encontrado)"
            echo ""
            echo "Scripts shell:"
            ls *.sh
            ;;
        5)
            echo ""
            echo "💾 FAZENDO BACKUP..."
            echo "══════════════════════════════════"
            if [ -f ~/backup_trinity.sh ]; then
                bash ~/backup_trinity.sh
            else
                echo "Criando backup manual..."
                DATA=$(date +%Y%m%d_%H%M)
                BACKUP_FILE="/sdcard/Download/trinity_${DATA}.tar.gz"
                tar -czf "$BACKUP_FILE" .
                echo "✅ Backup criado: $BACKUP_FILE"
            fi
            ;;
        0)
            echo ""
            echo "👋 Até logo, astronauta!"
            echo ""
            exit 0
            ;;
    esac
    
    echo ""
    echo "══════════════════════════════════"
    read -p "Pressione Enter para continuar..." wait
done
