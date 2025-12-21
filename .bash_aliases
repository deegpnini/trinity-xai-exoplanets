# ==========================================
# 📦 SISTEMA & PACOTES
# ==========================================
alias update='pkg update && pkg upgrade'
alias install='pkg install'
alias pkg-list='pkg list-installed'
alias pkg-search='pkg search'
alias pkg-remove='pkg remove'

# Utilitários Termux
alias storage='termux-setup-storage'
alias battery='termux-battery-status'
alias vibrate='termux-vibrate -d 100'
alias notify='termux-notification -t "Termux" -c'

# ==========================================
# 🔧 GIT & VERSIONAMENTO
# ==========================================
alias gs='git status'
alias ga='git add'
alias gaa='git add --all'
alias gc='git commit -m'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gpl='git pull'
alias gps='git push'

# Funções Git avançadas
function gl() {
    git log --oneline --graph --decorate --all -n 20
}

function gclean() {
    git fetch --prune
    git branch -vv | grep ': gone]' | awk '{print $1}' | xargs git branch -D
}

function gupdate() {
    git pull --rebase
}

function gcommit() {
    if [ -z "$1" ]; then
        echo "Uso: gcommit \"mensagem\""
        return 1
    fi
    git add --all
    git commit -m "$1"
}

function gpush() {
    branch=$(git branch --show-current)
    git push -u origin "$branch"
}

# ==========================================
# 🐍 PYTHON & DESENVOLVIMENTO
# ==========================================
alias py='python3'
alias pipi='pip install --no-binary :all:'
alias pyclean='find . -type f -name "*.py[co]" -delete -o -type d -name "__pycache__" -delete'

function venv() {
    python3 -m venv venv
    echo "Virtual environment criado. Ative com: source venv/bin/activate"
}

# ==========================================
# 📁 PROJETOS & ESTRUTURA
# ==========================================
alias ll='ls -la'
alias ..='cd ..'
alias ...='cd ../..'
alias c='clear'

function mkcd() {
    if [ -z "$1" ]; then
        echo "Uso: mkcd <nome_da_pasta>"
        return 1
    fi
    mkdir -p "$1" && cd "$1"
    echo "📁 Criado e entrou em: $PWD"
}

function backup() {
    if [ -z "$1" ]; then
        dir_name=$(basename "$PWD")
        backup_name="${dir_name}_backup_$(date +%Y%m%d_%H%M%S)"
    else
        backup_name="$1"
    fi
    
    echo "📦 Criando backup: $backup_name.tar.gz"
    tar -czf ~/backups/"$backup_name".tar.gz .
    echo "✅ Backup salvo em: ~/backups/$backup_name.tar.gz"
}

function ds() {
    du -sh ./* | sort -hr
}

# ==========================================
# 🛠️ UTILITÁRIOS & FERRAMENTAS
# ==========================================
function myip() {
    curl -s ifconfig.me
    echo
}

function weather() {
    if [ -z "$1" ]; then
        city="São Paulo"
    else
        city="$1"
    fi
    curl -s "wttr.in/$city?format=3"
}

function calc() {
    python3 -c "print($1)"
}

function extract() {
    if [ -f "$1" ]; then
        case $1 in
            *.tar.bz2)   tar xjf "$1"     ;;
            *.tar.gz)    tar xzf "$1"     ;;
            *.bz2)       bunzip2 "$1"     ;;
            *.rar)       unrar x "$1"     ;;
            *.gz)        gunzip "$1"      ;;
            *.tar)       tar xf "$1"      ;;
            *.tbz2)      tar xjf "$1"     ;;
            *.tgz)       tar xzf "$1"     ;;
            *.zip)       unzip "$1"       ;;
            *.Z)         uncompress "$1"  ;;
            *.7z)        7z x "$1"        ;;
            *)           echo "'$1' não pode ser extraído via extract()" ;;
        esac
    else
        echo "'$1' não é um arquivo válido"
    fi
}

function ff() {
    find . -type f -name "*$1*" 2>/dev/null
}

function fd() {
    find . -type d -name "*$1*" 2>/dev/null
}

alias ping='ping -c 4'

# ==========================================
# 🚀 TRINITY FALCON LUNG
# ==========================================
alias trinity='cd ~/trinity_falcon_lung'
alias start-trinity='cd ~/trinity_falcon_lung && ./start_trinity.sh'

function trinity-backup() {
    echo "🚀 Criando backup do Trinity Falcon Lung..."
    cd ~/trinity_falcon_lung
    backup_name="trinity_backup_$(date +%Y%m%d_%H%M%S)"
    tar -czf ~/backups/"$backup_name".tar.gz .
    echo "✅ Backup criado: ~/backups/$backup_name.tar.gz"
}

# ==========================================
# 🆘 AJUDA E MENUS
# ==========================================
alias help-menu='bash ~/scripts/ajuda.sh'
alias quick-help='bash ~/scripts/help_quick.sh'

function commands() {
    echo "🚀 COMANDOS DISPONÍVEIS:"
    echo "========================"
    echo "📦 Sistema: update, install, storage, battery"
    echo "🔧 Git: gs, ga, gc, gco, gl, gclean, gpush"
    echo "🐍 Python: py, pipi, pyclean, venv"
    echo "📁 Projetos: mkcd, backup, ll, ds"
    echo "🛠️ Utilitários: myip, weather, calc, extract"
    echo "🚀 Trinity: trinity, start-trinity, trinity-backup"
    echo "❓ Ajuda: help-menu, quick-help, commands"
    echo ""
}
