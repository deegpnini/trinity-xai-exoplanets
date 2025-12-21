#!/bin/bash

show_header() {
    clear
    echo "=========================================="
    echo "🆘 MENU DE AJUDA - TERMUX PROFISSIONAL"
    echo "=========================================="
    echo ""
}

show_system() {
    echo "📦 SISTEMA & PACOTES"
    echo "===================="
    echo ""
    echo "COMANDOS DISPONÍVEIS:"
    echo "  update              - Atualizar todos os pacotes"
    echo "  install <pacote>    - Instalar pacote"
    echo "  pkg-list            - Listar pacotes instalados"
    echo "  pkg-search <termo>  - Procurar pacote"
    echo "  pkg-remove <pacote> - Remover pacote"
    echo ""
    echo "UTILITÁRIOS TERMUX:"
    echo "  storage             - Acessar armazenamento"
    echo "  battery             - Status da bateria"
    echo "  vibrate             - Vibrar celular (100ms)"
    echo "  notify \"mensagem\"   - Enviar notificação"
    echo ""
}

show_git() {
    echo "🔧 GIT & VERSIONAMENTO"
    echo "======================"
    echo ""
    echo "COMANDOS BÁSICOS:"
    echo "  gs                  - git status"
    echo "  ga <arquivo>        - git add"
    echo "  gaa                 - git add --all"
    echo "  gc \"mensagem\"      - git commit -m"
    echo "  gco <branch>        - git checkout"
    echo "  gcb <branch>        - git checkout -b (nova branch)"
    echo "  gpl                 - git pull"
    echo "  gps                 - git push"
    echo ""
    echo "COMANDOS AVANÇADOS:"
    echo "  gl                  - git log formatado"
    echo "  gclean              - Limpar branches antigos"
    echo "  gupdate             - Atualizar com rebase"
    echo "  gcommit \"msg\"      - Add + Commit automático"
    echo "  gpush               - Push para branch atual"
    echo ""
}

show_python() {
    echo "🐍 PYTHON & DESENVOLVIMENTO"
    echo "==========================="
    echo ""
    echo "COMANDOS PYTHON:"
    echo "  py <script.py>      - Executar script Python"
    echo "  pipi <pacote>       - Instalar sem binary"
    echo "  pyclean             - Limpar cache Python"
    echo "  venv                - Criar virtual environment"
    echo ""
    echo "BIBLIOTECAS INSTALADAS:"
    pip list | grep -E "requests|flask|numpy|ipython|pytest|pylint|black|rich"
    echo ""
}

show_projects() {
    echo "📁 PROJETOS & ESTRUTURA"
    echo "======================"
    echo ""
    echo "COMANDOS:"
    echo "  mkcd <nome>         - Criar pasta e entrar"
    echo "  backup [nome]       - Backup do diretório"
    echo "  ll                  - Listar detalhado"
    echo "  ds                  - Tamanho de diretórios"
    echo "  ..                  - Voltar um nível"
    echo "  ...                 - Voltar dois níveis"
    echo "  c                   - Limpar tela"
    echo ""
    echo "ESTRUTURA:"
    echo "  ~/projects/         - Todos os projetos"
    echo "  ~/scripts/          - Scripts úteis"
    echo "  ~/backups/          - Backups"
    echo "  ~/tmp/              - Temporários"
    echo ""
}

show_utils() {
    echo "🛠️ UTILITÁRIOS & FERRAMENTAS"
    echo "============================"
    echo ""
    echo "SISTEMA:"
    echo "  myip                - Ver IP público"
    echo "  weather [cidade]    - Ver clima"
    echo "  calc \"expressão\"    - Calculadora"
    echo ""
    echo "ARQUIVOS:"
    echo "  extract <arquivo>   - Extrair qualquer arquivo"
    echo "  ff <nome>           - Encontrar arquivo"
    echo "  fd <nome>           - Encontrar diretório"
    echo ""
    echo "REDE:"
    echo "  ping <host>         - Testar conexão"
    echo "  curl <url>          - Requisição HTTP"
    echo ""
}

show_trinity() {
    echo "🚀 TRINITY FALCON LUNG"
    echo "======================"
    echo ""
    echo "COMANDOS:"
    echo "  trinity             - Ir para diretório"
    echo "  start-trinity       - Iniciar menu"
    echo "  trinity-backup      - Backup do projeto"
    echo ""
    echo "LOCALIZAÇÃO:"
    echo "  ~/trinity_falcon_lung/"
    echo ""
    if [ -d ~/trinity_falcon_lung ]; then
        echo "CONTEÚDO DO DIRETÓRIO:"
        ls -la ~/trinity_falcon_lung/
        echo ""
    fi
}

# Menu principal
while true; do
    show_header
    
    echo "🎯 CATEGORIAS DE COMANDOS:"
    echo ""
    echo "1️⃣  SISTEMA & PACOTES"
    echo "2️⃣  GIT & VERSIONAMENTO"
    echo "3️⃣  PYTHON & DESENVOLVIMENTO"
    echo "4️⃣  PROJETOS & ESTRUTURA"
    echo "5️⃣  UTILITÁRIOS & FERRAMENTAS"
    echo "6️⃣  TRINITY FALCON LUNG"
    echo "7️⃣  TESTAR COMANDOS"
    echo "8️⃣  SAIR"
    echo ""
    
    read -p "Escolha uma opção [1-8]: " option
    
    case $option in
        1)
            show_header
            show_system
            ;;
        2)
            show_header
            show_git
            ;;
        3)
            show_header
            show_python
            ;;
        4)
            show_header
            show_projects
            ;;
        5)
            show_header
            show_utils
            ;;
        6)
            show_header
            show_trinity
            ;;
        7)
            show_header
            echo "🧪 TESTANDO COMANDOS..."
            echo ""
            echo "1. Testando git status..."
            gs 2>/dev/null || echo "  (não é um repositório git)"
            echo ""
            echo "2. Testando lista de arquivos..."
            ll
            echo ""
            echo "3. Testando IP público..."
            myip
            echo ""
            echo "✅ Comandos testados!"
            ;;
        8)
            echo "Saindo..."
            exit 0
            ;;
        *)
            echo "Opção inválida!"
            ;;
    esac
    
    echo ""
    echo "=========================================="
    read -p "Pressione Enter para continuar..."
done
