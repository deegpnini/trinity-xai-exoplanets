#!/bin/bash

PROJECTS_DIR="$HOME/projects"

list_projects() {
    echo "📁 SEUS PROJETOS:"
    echo ""
    
    for category in personal work opensource learning experiments; do
        if [ -d "$PROJECTS_DIR/$category" ] && [ "$(ls -A "$PROJECTS_DIR/$category")" ]; then
            echo "📂 $category/:"
            for project in "$PROJECTS_DIR/$category"/*; do
                if [ -d "$project" ]; then
                    project_name=$(basename "$project")
                    size=$(du -sh "$project" 2>/dev/null | cut -f1)
                    echo "   • $project_name ($size)"
                fi
            done
            echo ""
        fi
    done
}

create_project() {
    if [ -z "$1" ]; then
        echo "Uso: create_project <nome> [categoria]"
        echo "Categorias: personal, work, opensource, learning, experiments"
        return 1
    fi
    
    CATEGORY="${2:-personal}"
    PROJECT_PATH="$PROJECTS_DIR/$CATEGORY/$1"
    
    if [ -d "$PROJECT_PATH" ]; then
        echo "⚠️  Projeto '$1' já existe em $CATEGORY/"
        return 1
    fi
    
    mkdir -p "$PROJECT_PATH"
    cd "$PROJECT_PATH"
    
    # Inicializar Git
    git init
    
    # Criar estrutura básica
    mkdir -p src tests docs data
    
    # Criar README
    cat > README.md << README
# $1

## Descrição
[Descreva seu projeto aqui]

## Tecnologias
- 

## Como executar
\`\`\`bash
# Instalação
pip install -r requirements.txt

# Execução
python src/main.py
\`\`\`

## Estrutura
\`\`\`
$1/
├── src/           # Código fonte
├── tests/         # Testes
├── docs/          # Documentação
├── data/          # Dados
├── README.md      # Este arquivo
└── requirements.txt # Dependências
\`\`\`

## Autor
Hebron (@deegpnini)
README
    
    # Criar requirements.txt
    echo "# Dependências do projeto" > requirements.txt
    
    # Criar .gitignore
    cat > .gitignore << GITIGNORE
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Environments
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
GITIGNORE
    
    echo "🎉 Projeto '$1' criado em: $PROJECT_PATH"
    echo "📁 Estrutura criada: src/, tests/, docs/, data/"
    echo "📄 Arquivos: README.md, .gitignore, requirements.txt"
    echo ""
    echo "🚀 Para começar:"
    echo "   cd $PROJECT_PATH"
    echo "   git add . && git commit -m 'Initial commit'"
}

delete_project() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Uso: delete_project <nome> <categoria>"
        return 1
    fi
    
    PROJECT_PATH="$PROJECTS_DIR/$2/$1"
    
    if [ ! -d "$PROJECT_PATH" ]; then
        echo "❌ Projeto '$1' não encontrado em $2/"
        return 1
    fi
    
    read -p "⚠️  Tem certeza que deseja excluir '$1'? (s/n): " confirm
    
    if [[ "$confirm" == "s" || "$confirm" == "S" ]]; then
        rm -rf "$PROJECT_PATH"
        echo "✅ Projeto '$1' excluído"
    else
        echo "❌ Exclusão cancelada"
    fi
}

backup_projects() {
    BACKUP_FILE="$HOME/backups/projects_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    echo "💾 Criando backup de todos os projetos..."
    cd "$HOME"
    tar -czf "$BACKUP_FILE" projects/
    
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "✅ Backup criado: $BACKUP_FILE ($SIZE)"
}

# Menu principal
case "$1" in
    list)
        list_projects
        ;;
    create)
        create_project "$2" "$3"
        ;;
    delete)
        delete_project "$2" "$3"
        ;;
    backup)
        backup_projects
        ;;
    *)
        echo "📋 PROJECT MANAGER"
        echo ""
        echo "Uso: project <comando>"
        echo ""
        echo "Comandos disponíveis:"
        echo "  list                          Listar todos os projetos"
        echo "  create <nome> [categoria]     Criar novo projeto"
        echo "  delete <nome> <categoria>     Excluir projeto"
        echo "  backup                        Backup de todos os projetos"
        echo ""
        echo "Categorias: personal, work, opensource, learning, experiments"
        ;;
esac
