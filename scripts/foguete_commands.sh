#!/bin/bash

# Comandos rápidos para o projeto Foguete Híbrido BR
alias foguete='cd ~/projects/foguete_hibrido'
alias simular='cd ~/projects/foguete_hibrido/v5_hybrid && python simulacao_completa.py'
alias pitch='cd ~/projects/foguete_hibrido/pitch && python investor_pitch.py'
alias backup-foguete='cd ~/projects/foguete_hibrido && ./backup_projeto.sh'

function foguete-status() {
    echo "🚀 STATUS DO FOGUETE HÍBRIDO BR"
    echo "================================"
    echo ""
    echo "📊 ÚLTIMOS RESULTADOS (v5.0):"
    echo "  • Altitude: 36km"
    echo "  • Combustível restante: 47.75t"
    echo "  • Energia eólica: 35.32 kWh"
    echo "  • Redução CO₂: 325kg"
    echo "  • Economia: R$350k"
    echo ""
    echo "🎯 PRÓXIMOS PASSOS:"
    echo "  1. Otimização eólica (10-20m/s)"
    echo "  2. Configuração multi-estágio"
    echo "  3. Aumento de payload (+180kg)"
    echo "  4. Busca por investidores"
    echo ""
    echo "🔧 COMANDOS DISPONÍVEIS:"
    echo "  foguete         - Ir para diretório do projeto"
    echo "  simular         - Rodar simulação completa"
    echo "  pitch           - Gerar pitch para investidores"
    echo "  backup-foguete  - Backup completo do projeto"
    echo "  foguete-status  - Ver este relatório"
    echo ""
}

function foguete-otimizar() {
    echo "🌪️ OTIMIZAÇÃO EÓLICA - CÁLCULO RÁPIDO"
    echo "====================================="
    echo ""
    echo "Dados atuais: 35.32 kWh"
    echo ""
    
    # Cálculos rápidos
    echo "Para 15 m/s:"
    echo "  • +12.5% de energia"
    echo "  • Energia total: ~39.7 kWh"
    echo "  • Economia adicional: ~R$43.7k"
    echo ""
    
    echo "Para 20 m/s:"
    echo "  • +33.3% de energia"
    echo "  • Energia total: ~47.1 kWh"
    echo "  • Economia adicional: ~R$116.5k"
    echo ""
    
    echo "💰 ECONOMIA TOTAL POTENCIAL:"
    echo "  • Com 15 m/s: R$393.7k"
    echo "  • Com 20 m/s: R$466.5k"
    echo ""
}
