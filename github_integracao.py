#!/usr/bin/env python3
"""
SISTEMA DE INTEGRAÇÃO GITHUB + PROJETO INTERESTELAR
Autor: Helyton R. G. Ronchi (Hebron)
GitHub: deegpnini
Repositório: trinity-xai-exoplanets
"""

import os
import sys
import json
from datetime import datetime
import subprocess

class GitHubIntegrador:
    def __init__(self):
        self.autor = "Helyton R. G. Ronchi (Hebron)"
        self.github_user = "deegpnini"
        self.repositorio = "trinity-xai-exoplanets"
        self.data = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        # Arquivos do projeto interestelar
        self.arquivos_projeto = [
            "teste_simples.py",
            "sistema_basico.py", 
            "sistema_interativo.py",
            "sistema_interestelar.py",
            "sistema_api_real.py",
            "sistema_interestelar_final.py",
            "sistema_conexoes_corrigido.py",
            "github_integracao.py"
        ]
        
        # Estrutura do repositório
        self.estrutura = {
            "projeto_interestelar/": [
                "README.md",
                "scripts/",
                "dados/",
                "simulacoes/",
                "apis/",
                "documentacao/"
            ],
            "trinity_framework/": [
                "README.md",
                "ai_models/",
                "astronomy_data/",
                "collaborative_systems/"
            ]
        }
    
    def cabecalho(self):
        print("="*70)
        print("🚀 INTEGRAÇÃO GITHUB - PROJETO INTERESTELAR")
        print("="*70)
        print(f"👤 GitHub User: {self.github_user}")
        print(f"📁 Repositório: {self.repositorio}")
        print(f"👨‍💻 Autor: {self.autor}")
        print(f"📅 Data: {self.data}")
        print("="*70)
    
    def verificar_git_instalado(self):
        """Verifica se Git está instalado"""
        print("\n🔍 VERIFICANDO INSTALAÇÃO DO GIT...")
        try:
            resultado = subprocess.run(["git", "--version"], 
                                     capture_output=True, text=True)
            if resultado.returncode == 0:
                print(f"✅ Git instalado: {resultado.stdout.strip()}")
                return True
            else:
                print("❌ Git não encontrado")
                return False
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def verificar_arquivos_locais(self):
        """Verifica quais arquivos do projeto existem localmente"""
        print("\n📁 ARQUIVOS DO PROJETO INTERESTELAR:")
        print("-"*40)
        
        existentes = []
        faltantes = []
        
        for arquivo in self.arquivos_projeto:
            if os.path.exists(arquivo):
                tamanho = os.path.getsize(arquivo)
                existentes.append((arquivo, tamanho))
                print(f"  ✅ {arquivo} ({tamanho} bytes)")
            else:
                faltantes.append(arquivo)
                print(f"  ❌ {arquivo} (faltando)")
        
        return existentes, faltantes
    
    def criar_readme_projeto(self):
        """Cria README para o projeto interestelar"""
        readme_content = f"""# Projeto Interestelar Brasileiro

## Autor: Helyton R. G. Ronchi (Hebron)
## GitHub: {self.github_user}
## Data de início: 21/12/2024
## Status: Em desenvolvimento

## 🎯 OBJETIVO
Desenvolver sistema para análise in situ de objetos interestelares
usando tecnologia brasileira (LIBS em foguete suborbital).

## 📡 FOCO EM 3I/ATLAS
- Objeto interestelar a ser descoberto em 2025
- Aproximação máxima da Terra: 19/12/2025
- Nosso alvo: Primeira análise espectral próxima

## 🛠️ TECNOLOGIA
- LIBS (Laser-Induced Breakdown Spectroscopy)
- Foguete suborbital híbrido
- Sistema de resposta rápida (30-60 dias)

## 📂 ESTRUTURA DO PROJETO

### scripts/
- Sistema básico de simulação
- Integração com APIs (NASA, ESA)
- Análise de dados

### dados/
- Informações sobre objetos interestelares
- Simulações orbitais
- Resultados esperados

### simulacoes/
- Trajetórias de rendezvous
- Espectros LIBS simulados
- Análise elementar

### apis/
- Conexão com APIs astronômicas
- Processamento de dados em tempo real
- Integração com observatórios

## 🚀 CRONOGRAMA
- 2024: Desenvolvimento do sistema
- 2025: Testes com 3I/ATLAS (simulado)
- 2026: Sistema operacional completo
- 2027+: Pronto para próximo objeto interestelar

## 👥 COLABORAÇÃO
Projeto aberto para colaboração com:
- Observatórios brasileiros
- Universidades (INPE, ITA, USP)
- Pesquisadores independentes
- Entusiastas da astronomia

## 📞 CONTATO
- GitHub: {self.github_user}
- Email: deegp.nini@gmail.com
- LinkedIn: https://www.linkedin.com/in/helyton-gonçalves-renato-ronch-606188392

## 📄 LICENÇA
Código aberto para fins educacionais e de pesquisa.

---
*Desenvolvido com paixão pela exploração espacial brasileira.*
"""
        
        with open("README_INTERESTELAR.md", "w") as f:
            f.write(readme_content)
        
        print(f"\n📄 README criado: README_INTERESTELAR.md")
        print(f"   Tamanho: {len(readme_content)} bytes")
        
        return readme_content
    
    def criar_estrutura_repositorio(self):
        """Cria estrutura de pastas para o repositório"""
        print("\n📁 CRIANDO ESTRUTURA DO REPOSITÓRIO...")
        print("-"*40)
        
        pastas_principais = ["projeto_interestelar", "trinity_framework"]
        
        for pasta in pastas_principais:
            os.makedirs(pasta, exist_ok=True)
            print(f"  ✅ Criada pasta: {pasta}/")
            
            # Subpastas
            subpastas = self.estrutura.get(f"{pasta}/", [])
            for subpasta in subpastas:
                if subpasta.endswith("/"):
                    caminho = os.path.join(pasta, subpasta)
                    os.makedirs(caminho, exist_ok=True)
                    print(f"    📁 {caminho}")
                else:
                    caminho = os.path.join(pasta, subpasta)
                    with open(caminho, "w") as f:
                        f.write(f"# {subpasta}\n\nArquivo do projeto {pasta}")
                    print(f"    📄 {caminho}")
    
    def gerar_relatorio_projeto(self):
        """Gera relatório completo do projeto"""
        print("\n📊 GERANDO RELATÓRIO DO PROJETO...")
        print("-"*40)
        
        relatorio = {
            "metadata": {
                "autor": self.autor,
                "github_user": self.github_user,
                "data_geracao": self.data,
                "projeto": "Interestelar Brasileiro",
                "versao": "1.0"
            },
            "arquivos_criados": self.arquivos_projeto,
            "estrutura_repositorio": self.estrutura,
            "status": {
                "git_instalado": self.verificar_git_instalado(),
                "arquivos_locais": len([a for a in self.arquivos_projeto if os.path.exists(a)]),
                "total_arquivos": len(self.arquivos_projeto)
            },
            "proximos_passos": [
                "Configurar Git localmente",
                "Conectar com repositório GitHub existente",
                "Fazer primeiro commit",
                "Subir arquivos do projeto interestelar",
                "Adicionar colaboradores",
                "Configurar GitHub Actions para automação"
            ]
        }
        
        # Salvar relatório
        with open("relatorio_projeto.json", "w") as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Relatório salvo: relatorio_projeto.json")
        
        # Mostrar resumo
        print(f"\n📈 RESUMO DO PROJETO:")
        print(f"  • Autor: {relatorio['metadata']['autor']}")
        print(f"  • Arquivos criados: {relatorio['status']['arquivos_locais']}/{relatorio['status']['total_arquivos']}")
        print(f"  • Git instalado: {'Sim' if relatorio['status']['git_instalado'] else 'Não'}")
        
        return relatorio
    
    def instrucoes_git_push(self):
        """Mostra instruções para subir para GitHub"""
        print("\n" + "="*70)
        print("📤 INSTRUÇÕES PARA SUBIR PARA GITHUB:")
        print("="*70)
        
        instrucoes = f"""
        PASSO 1: CONFIGURAR GIT LOCALMENTE
        ----------------------------------
        git config --global user.name "{self.autor}"
        git config --global user.email "deegp.nini@gmail.com"
        
        PASSO 2: INICIALIZAR REPOSITÓRIO
        --------------------------------
        cd /caminho/para/seu/projeto
        git init
        
        PASSO 3: CONECTAR COM SEU REPOSITÓRIO EXISTENTE
        ------------------------------------------------
        git remote add origin https://github.com/{self.github_user}/{self.repositorio}.git
        
        PASSO 4: ADICIONAR ARQUIVOS
        ---------------------------
        git add .
        git commit -m "Primeiro commit: Projeto Interestelar Brasileiro por {self.autor}"
        
        PASSO 5: SUBIR PARA GITHUB
        --------------------------
        git branch -M main
        git push -u origin main
        
        PASSO 6: VERIFICAR
        ------------------
        Acesse: https://github.com/{self.github_user}/{self.repositorio}
        """
        
        print(instrucoes)
        
        # Salvar instruções em arquivo
        with open("INSTRUCOES_GITHUB.txt", "w") as f:
            f.write(instrucoes)
        
        print(f"\n📄 Instruções salvas em: INSTRUCOES_GITHUB.txt")
    
    def modo_interativo(self):
        """Modo interativo do sistema GitHub"""
        print("\n" + "="*70)
        print("🤖 MODO INTERATIVO - GITHUB INTEGRATION")
        print("="*70)
        
        while True:
            print("\nComandos disponíveis:")
            print("  1. status    - Verificar arquivos e Git")
            print("  2. readme    - Criar README do projeto")
            print("  3. estrutura - Criar pastas do repositório")
            print("  4. relatorio - Gerar relatório completo")
            print("  5. instrucoes - Como subir para GitHub")
            print("  6. sair      - Encerrar")
            
            cmd = input("\n🎯 Digite comando: ").strip().lower()
            
            if cmd in ["sair", "exit", "6"]:
                print("\n👋 Até logo! Continue desenvolvendo seu projeto!")
                break
            
            elif cmd in ["status", "1"]:
                self.verificar_git_instalado()
                existentes, faltantes = self.verificar_arquivos_locais()
                print(f"\n📊 Status: {len(existentes)} arquivos presentes, {len(faltantes)} faltando")
            
            elif cmd in ["readme", "2"]:
                self.criar_readme_projeto()
            
            elif cmd in ["estrutura", "3"]:
                self.criar_estrutura_repositorio()
            
            elif cmd in ["relatorio", "4"]:
                self.gerar_relatorio_projeto()
            
            elif cmd in ["instrucoes", "5"]:
                self.instrucoes_git_push()
            
            else:
                print(f"\n❌ Comando '{cmd}' não reconhecido")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    sistema = GitHubIntegrador()
    sistema.cabecalho()
    
    print("\n" + "="*70)
    print("🎯 PROJETO INTERESTELAR + GITHUB INTEGRATION")
    print("="*70)
    print(f"👤 Seu GitHub: {sistema.github_user}")
    print(f"📁 Repositório alvo: {sistema.repositorio}")
    print(f"💡 Você já tem {len(sistema.arquivos_projeto)} scripts do projeto!")
    print("="*70)
    
    # Verificações iniciais
    git_ok = sistema.verificar_git_instalado()
    existentes, faltantes = sistema.verificar_arquivos_locais()
    
    print("\n" + "="*70)
    print("📊 STATUS INICIAL:")
    print("="*70)
    
    if git_ok:
        print("✅ Git está INSTALADO - pronto para versionamento!")
    else:
        print("⚠️ Git NÃO instalado - instale com: pkg install git")
    
    print(f"✅ Arquivos do projeto: {len(existentes)}/{len(sistema.arquivos_projeto)}")
    
    if len(faltantes) > 0:
        print(f"⚠️ Faltando: {', '.join(faltantes[:3])}{'...' if len(faltantes) > 3 else ''}")
    
    # Criar README automaticamente
    print("\n📄 Criando README do projeto...")
    sistema.criar_readme_projeto()
    
    # Criar estrutura
    print("\n📁 Criando estrutura de pastas...")
    sistema.criar_estrutura_repositorio()
    
    # Gerar relatório
    print("\n📊 Gerando relatório completo...")
    relatorio = sistema.gerar_relatorio_projeto()
    
    print("\n" + "="*70)
    print("🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    print("="*70)
    print("1. Instalar Git (se não tiver): pkg install git")
    print("2. Configurar suas credenciais GitHub")
    print("3. Seguir instruções em: INSTRUCOES_GITHUB.txt")
    print("4. Subir seu projeto para GitHub")
    print("5. Compartilhar com colaboradores")
    print("="*70)
    
    # Modo interativo
    sistema.modo_interativo()
    
    print("\n" + "="*70)
    print("✅ SISTEMA DE INTEGRAÇÃO GITHUB CONCLUÍDO!")
    print("="*70)
    print(f"👤 Autor: {sistema.autor}")
    print(f"📅 Data: {sistema.data}")
    print(f"🌟 Seu projeto está pronto para versionamento!")
    print("="*70)
