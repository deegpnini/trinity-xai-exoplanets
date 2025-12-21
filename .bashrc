
# ==========================================
# 🐍 VIRTUALENVWRAPPER CONFIGURAÇÃO
# ==========================================
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/data/data/com.termux/files/usr/bin/python3
source /data/data/com.termux/files/usr/bin/virtualenvwrapper.sh

# Funções úteis para Python
pyclean() {
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
}

pyenv-create() {
    if [ -z "$1" ]; then
        echo "Uso: pyenv-create <nome-do-ambiente>"
        return 1
    fi
    mkvirtualenv "$1" --python=python3
}

pyenv-list() {
    workon
}

pyenv-activate() {
    if [ -z "$1" ]; then
        echo "Uso: pyenv-activate <nome-do-ambiente>"
        return 1
    fi
    workon "$1"
}

pyenv-deactivate() {
    deactivate
}

# ==========================================
# 🔧 GIT ALIASES AVANÇADOS
# ==========================================
alias gs='git status'
alias ga='git add'
alias gaa='git add --all'
alias gc='git commit -m'
alias gca='git commit --amend'
alias gcm='git commit -m'
alias gcam='git commit -am'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gb='git branch'
alias gbd='git branch -d'
alias gbD='git branch -D'
alias gbl='git branch --list'
alias gm='git merge'
alias gpl='git pull'
alias gps='git push'
alias gpf='git push --force-with-lease'
alias gcl='git clone'
alias gfp='git fetch --prune'
alias gr='git remote -v'
alias gra='git remote add'
alias grr='git remote remove'
alias gl='git log --oneline --graph --all'
alias gld='git log --oneline --graph --all --decorate'
alias gd='git diff'
alias gdc='git diff --cached'
alias gsh='git stash'
alias gshl='git stash list'
alias gsha='git stash apply'
alias gshp='git stash pop'
alias gshc='git stash clear'
alias grs='git restore'
alias grss='git restore --staged'
alias grb='git rebase'
alias grbi='git rebase -i'
alias grbc='git rebase --continue'
alias grba='git rebase --abort'

# Funções Git avançadas
gclean() {
    git fetch --prune
    git branch --merged | grep -v "\*" | grep -v "main" | grep -v "master" | xargs -n 1 git branch -d
    echo "✅ Branches antigos removidos"
}

gupdate() {
    current_branch=$(git branch --show-current)
    git checkout main 2>/dev/null || git checkout master
    git pull
    git checkout "$current_branch"
    git rebase main 2>/dev/null || git rebase master
    echo "✅ Repositório atualizado e rebaseado"
}

gcommit() {
    if [ -z "$1" ]; then
        echo "Uso: gcommit \"mensagem do commit\""
        return 1
    fi
    git add .
    git commit -m "$1"
    echo "✅ Commit realizado: $1"
}

gpush() {
    current_branch=$(git branch --show-current)
    git push origin "$current_branch"
    echo "✅ Push realizado para: $current_branch"
}

glog() {
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --date=relative
}
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# ==========================================
# 🚀 PROMPT PROFISSIONAL PARA TERMUX
# ==========================================

# Cores
RED='\[\033[0;31m\]'
GREEN='\[\033[0;32m\]'
YELLOW='\[\033[1;33m\]'
BLUE='\[\033[0;34m\]'
PURPLE='\[\033[0;35m\]'
CYAN='\[\033[0;36m\]'
WHITE='\[\033[1;37m\]'
NC='\[\033[0m\]' # No Color

# Git functions for prompt
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

parse_git_status() {
    local status=$(git status --porcelain 2>/dev/null)
    if [[ -n "$status" ]]; then
        echo "*"
    fi
}

# Prompt principal
PS1="${GREEN}┌─[${CYAN}\u${GREEN}]─[${YELLOW}\$(date +%H:%M:%S)${GREEN}]─[${BLUE}\w${GREEN}]"
PS1+="\$(if [ -n \"\$(parse_git_branch)\" ]; then echo \"─[${PURPLE}\$(parse_git_branch)${RED}\$(parse_git_status)${GREEN}]\"; fi)"
PS1+="\n${GREEN}└─[${WHITE}\$${GREEN}]${NC} "

# Prompt secundário (continuation)
PS2="${GREEN}└─[${WHITE}>${GREEN}]${NC} "

# Aliases úteis
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias c='clear'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias h='history'
alias j='jobs -l'
alias path='echo -e ${PATH//:/\\n}'
alias now='date +"%T"'
alias nowdate='date +"%d-%m-%Y"'
alias ports='netstat -tulanp'
alias update='pkg update && pkg upgrade'
alias install='pkg install'
alias remove='pkg remove'
alias search='pkg search'

# Funções úteis
extract() {
    if [ -f "$1" ] ; then
        case $1 in
            *.tar.bz2)   tar xvjf "$1"    ;;
            *.tar.gz)    tar xvzf "$1"    ;;
            *.bz2)       bunzip2 "$1"     ;;
            *.rar)       unrar x "$1"     ;;
            *.gz)        gunzip "$1"      ;;
            *.tar)       tar xvf "$1"     ;;
            *.tbz2)      tar xvjf "$1"    ;;
            *.tgz)       tar xvzf "$1"    ;;
            *.zip)       unzip "$1"       ;;
            *.Z)         uncompress "$1"  ;;
            *.7z)        7z x "$1"        ;;
            *)           echo "'$1' não pode ser extraído via extract()" ;;
        esac
    else
        echo "'$1' não é um arquivo válido"
    fi
}

# Network
myip() {
    curl ifconfig.me
    echo
}

# Weather
weather() {
    curl wttr.in/"${1:-São Paulo}"
}

# Calculator
calc() {
    echo "$*" | bc -l
}

# Create directory and enter it
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# Find file
ff() {
    find . -type f -iname "*$1*"
}

# Find directory
fd() {
    find . -type d -iname "*$1*"
}

# Grep with colors
grp() {
    grep --color=always -i "$1" "${@:2}"
}

# Size of directories
ds() {
    du -sh * | sort -h
}

# Battery info
battery() {
    termux-battery-status
}

# Clipboard
copy() {
    cat "$1" | termux-clipboard-set
}

paste() {
    termux-clipboard-get
}

# Backup rápido do diretório atual
backup() {
    local name="backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "../$name" .
    echo "✅ Backup criado: ../$name"
    ls -lh "../$name"
}

# MENSAGEM DE BOAS-VINDAS
echo ""
echo "${GREEN}==========================================${NC}"
echo "${GREEN}🚀 TERMUX PROFISSIONAL CONFIGURADO${NC}"
echo "${GREEN}==========================================${NC}"
echo "${CYAN}Dispositivo: Galaxy A70 (SM-A705MN)${NC}"
echo "${CYAN}Usuário: $(whoami)${NC}"
echo "${CYAN}Data: $(date)${NC}"
echo "${GREEN}==========================================${NC}"
echo ""
echo "${YELLOW}📁 Comandos disponíveis:${NC}"
echo "  • update        - Atualizar pacotes"
echo "  • backup        - Backup rápido do diretório"
echo "  • myip          - Ver IP público"
echo "  • battery       - Status da bateria"
echo "  • pyenv-create  - Criar ambiente Python"
echo "  • gupdate       - Atualizar repositório Git"
echo ""
echo "${BLUE}Git configurado para: Hebron <deegp.nini@gmail.com>${NC}"
echo "${PURPLE}Chave SSH disponível em: ~/.ssh/id_ed25519.pub${NC}"
echo ""
alias project="bash ~/scripts/project-manager.sh"

# ==========================================
# 🛠️ CONFIGURAÇÕES DE COMPILAÇÃO TERMUX
# ==========================================

# Evitar compilações pesadas no Android
export PACKAGES_DIR=/data/data/com.termux/files/usr
export CARGO_BUILD_TARGET=aarch64-linux-android
export RUSTFLAGS="-C target-feature=-crt-static"

# Para Python packages
export NPY_BLAS_ORDER=
export NPY_LAPACK_ORDER=
export NPY_DISABLE_SVML=1
export NPY_NO_SMP=1

# Otimizações para Android
export CFLAGS="-O2 -pipe -fPIC"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-Wl,-rpath=$PACKAGES_DIR/lib -Wl,--enable-new-dtags"

# Pip config para evitar compilações desnecessárias
export PIP_NO_BUILD_ISOLATION=0
export PIP_PROGRESS_BAR=off

# Função para instalar pacotes sem compilação
pip-install-light() {
    echo "📦 Instalando versões leves..."
    
    # Primeiro tentar wheel pré-compilado
    pip install --no-deps --prefer-binary "$1" 2>/dev/null
    
    # Se falhar, instalar sem dependências de compilação
    if [ $? -ne 0 ]; then
        echo "⚠️  Usando instalação minimalista..."
        pip install --no-deps "$1"
    fi
}

# ==========================================
# 🚀 CONFIGURAÇÃO PROFISSIONAL TERMUX
# ==========================================

# Cores
RED='\[\033[0;31m\]'
GREEN='\[\033[0;32m\]'
YELLOW='\[\033[1;33m\]'
BLUE='\[\033[0;34m\]'
PURPLE='\[\033[0;35m\]'
CYAN='\[\033[0;36m\]'
WHITE='\[\033[1;37m\]'
NC='\[\033[0m\]'

# Git no prompt
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

# Prompt personalizado
PS1="${GREEN}┌─[${CYAN}\u${GREEN}]─[${YELLOW}\$(date +%H:%M:%S)${GREEN}]─[${BLUE}\w${GREEN}]"
PS1+="\$(if [ -n \"\$(parse_git_branch)\" ]; then echo \"─[${PURPLE}\$(parse_git_branch)${GREEN}]\"; fi)"
PS1+="\n${GREEN}└─[${WHITE}\$${GREEN}]${NC} "

# Aliases úteis
alias ll='ls -la'
alias la='ls -A'
alias c='clear'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias update='pkg update && pkg upgrade'
alias install='pkg install'
alias pipi='pip install --no-binary :all:'

# Funções úteis
mkcd() { mkdir -p "$1" && cd "$1"; }
backup() { tar -czf "../backup_$(date +%Y%m%d_%H%M%S).tar.gz" .; }
pyclean() { find . -type f -name "*.py[co]" -delete; find . -type d -name "__pycache__" -delete; }

# Mensagem de boas-vindas
echo ""
echo "${GREEN}==========================================${NC}"
echo "${GREEN}🚀 TERMUX PROFISSIONAL - PRONTO PARA CÓDIGO${NC}"
echo "${GREEN}==========================================${NC}"
echo "${CYAN}Usuário: $(whoami)${NC}"
echo "${CYAN}Python: $(python3 --version 2>/dev/null || echo 'Não instalado')${NC}"
echo "${CYAN}Git: $(git --version 2>/dev/null || echo 'Não instalado')${NC}"
echo "${GREEN}==========================================${NC}"
echo ""
alias help="bash ~/scripts/help_quick.sh"

# Carregar aliases personalizados
if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi


# Mensagem de boas-vindas
echo ""
echo "🎯 TERMUX PROFISSIONAL - COMANDOS DISPONÍVEIS"
echo "📦 Sistema: update, install, storage"
echo "🔧 Git: gs, ga, gc, gpush"
echo "🐍 Python: py, pipi, venv"
echo "📁 Projetos: mkcd, backup, ll"
echo "🚀 Trinity: trinity, start-trinity"
echo "❓ Ajuda: help-menu, commands"
echo ""

# Comandos do Foguete Híbrido BR
source ~/scripts/foguete_commands.sh
