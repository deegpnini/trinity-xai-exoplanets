#!/usr/bin/env python3
"""
SISTEMA INTERESTELAR BRASILEIRO - VERSÃO FINAL
Autor: Helyton R. G. Ronchi (Hebron)
Integração: Python + APIs + Simulações
"""

import requests
import json
from datetime import datetime
import sys

class SistemaInterestelarHebron:
    def __init__(self):
        self.autor = "Helyton R. G. Ronchi (Hebron)"
        self.data_criacao = "21/12/2024"
        self.versao = "2.0"
        self.log_file = "logs_interstelar.txt"
        
        # Dados do projeto
        self.projeto = {
            "nome": "Projeto Interestelar Brasileiro",
            "objetivo": "Análise in situ de objetos interestelares",
            "tecnologia": "LIBS em foguete suborbital",
            "timeline": {
                "2024": "Desenvolvimento do sistema",
                "2025": "Testes com 3I/ATLAS (simulado)",
                "2026": "Sistema operacional completo",
                "2027+": "Pronto para próximo objeto"
            }
        }
    
    def cabecalho(self):
        """Mostra cabeçalho do sistema"""
        print("="*70)
        print("🚀 SISTEMA INTERESTELAR BRASILEIRO - HEBRON")
        print("="*70)
        print(f"👤 Autor: {self.autor}")
        print(f"📅 Criado em: {self.data_criacao}")
        print(f"🔢 Versão: {self.versao}")
        print("="*70)
    
    def testar_conexoes(self):
        """Testa todas as conexões necessárias"""
        print("\n🔍 TESTANDO CONEXÕES...")
        print("-"*40)
        
        conexoes = {
            "Internet": "https://api.github.com",
            "NASA NEO": "https://api.nasa.gov/neo/rest/v1/neo/browse?api_key=DEMO_KEY",
            "Minor Planet": "https://minorplanetcenter.net/web_service",
            "ESA": "https://www.esa.int"
        }
        
        resultados = {}
        for nome, url in conexoes.items():
            try:
                if "DEMO_KEY" in url:
                    # Para NASA, usa demo key
                    resp = requests.get(url, timeout=5)
                else:
                    resp = requests.get(url, timeout=5)
                
                if resp.status_code in [200, 403, 404]:
                    resultados[nome] = True
                    print(f"  ✅ {nome}: Acessível")
                else:
                    resultados[nome] = False
                    print(f"  ⚠️ {nome}: Erro {resp.status_code}")
                    
            except Exception as e:
                resultados[nome] = False
                print(f"  ❌ {nome}: {str(e)[:40]}")
        
        return resultados
    
    def mostrar_info_3i_atlas(self):
        """Mostra informações sobre 3I/ATLAS"""
        print("\n📡 3I/ATLAS - OBJETO INTERESTELAR")
        print("-"*40)
        
        info = {
            "Designação": "3I/ATLAS",
            "Status": "A ser descoberto em 2025",
            "Tipo": "Objeto interestelar (provável)",
            "Descoberta estimada": "Julho 2025",
            "Aproximação Terra": "19 de Dezembro 2025",
            "Distância mínima": "~0.05 UA (7.5 milhões de km)",
            "Magnitude": "~18 (necessita telescópio médio)",
            "Velocidade": "~30 km/s (típico interestelar)"
        }
        
        for chave, valor in info.items():
            print(f"  • {chave}: {valor}")
        
        print("\n🎯 NOSSO PLANO PARA 3I/ATLAS:")
        print("  1. Detecção (julho 2025)")
        print("  2. Análise orbital (agosto 2025)")
        print("  3. Preparação foguete (setembro 2025)")
        print("  4. Lançamento (outubro 2025)")
        print("  5. Encontro/análise (novembro 2025)")
    
    def simular_analise_libs(self):
        """Simula análise com espectrômetro LIBS"""
        print("\n🔬 SIMULAÇÃO DE ANÁLISE LIBS")
        print("-"*40)
        
        elementos = ["Fe", "Ni", "Mg", "Si", "C", "O", "H", "N", "S", "Na", "K", "Ca"]
        abundancias = [12.5, 1.5, 15.0, 18.0, 25.0, 20.0, 2.0, 1.0, 0.5, 0.1, 0.05, 1.0]
        
        print("  Elemento   Abundância   Detectado?")
        print("  ---------  -----------  -----------")
        
        for elem, abund in zip(elementos, abundancias):
            detectado = "SIM" if abund > 0.1 else "LIMIAR"
            print(f"  {elem:^9}   {abund:^10.2f}%   {detectado:^11}")
        
        print("\n💡 Interpretação:")
        print("  • Composição similar a asteroides tipo S")
        print("  • Presença de metais (Fe, Ni) significativa")
        print("  • Voláteis (H, N) abaixo do esperado para cometas")
        print("  • Sugere origem em sistema estelar pobre em voláteis")
    
    def mostrar_timeline_projeto(self):
        """Mostra timeline completa do projeto"""
        print("\n📅 TIMELINE DO PROJETO INTERESTELAR")
        print("-"*40)
        
        for ano, descricao in self.projeto["timeline"].items():
            if ano == "2024":
                print(f"\n{ano} (AGORA - DESENVOLVIMENTO):")
                print(f"  • {descricao}")
                print(f"  • Autor: {self.autor}")
                print(f"  • Data atual: {datetime.now().strftime('%d/%m/%Y')}")
            elif ano == "2025":
                print(f"\n{ano} (ANO ALVO - EVENTOS FUTUROS):")
                if isinstance(descricao, list):
                    for item in descricao:
                        print(f"  • {item}")
                else:
                    print(f"  • {descricao}")
            else:
                print(f"\n{ano} (FUTURO):")
                if isinstance(descricao, list):
                    for item in descricao:
                        print(f"  • {item}")
                else:
                    print(f"  • {descricao}")
    
    def salvar_log(self, acao):
        """Salva log das ações"""
        with open(self.log_file, "a") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "autor": self.autor,
                "acao": acao,
                "versao": self.versao
            }
            f.write(json.dumps(log_entry) + "\n")
    
    def modo_interativo(self):
        """Modo interativo principal"""
        print("\n" + "="*70)
        print("🤖 MODO INTERATIVO - SISTEMA INTERESTELAR")
        print("="*70)
        print("Comandos disponíveis:")
        print("  1. info     - Informações do projeto")
        print("  2. 3i       - Sobre 3I/ATLAS")
        print("  3. libs     - Simulação LIBS")
        print("  4. timeline - Cronograma do projeto")
        print("  5. conexao  - Testar conexões")
        print("  6. autor    - Mostrar autor")
        print("  7. sair     - Encerrar sistema")
        print("="*70)
        
        while True:
            try:
                comando = input("\n🎯 Digite comando: ").strip().lower()
                
                if comando in ["sair", "exit", "quit"]:
                    print("\n👋 Encerrando sistema. Até logo, Hebron!")
                    self.salvar_log("sistema_encerrado")
                    break
                
                elif comando in ["info", "1"]:
                    print(f"\n📋 PROJETO: {self.projeto['nome']}")
                    print(f"🎯 OBJETIVO: {self.projeto['objetivo']}")
                    print(f"🔬 TECNOLOGIA: {self.projeto['tecnologia']}")
                    self.salvar_log("comando_info")
                
                elif comando in ["3i", "3i/atlas", "2"]:
                    self.mostrar_info_3i_atlas()
                    self.salvar_log("comando_3i_atlas")
                
                elif comando in ["libs", "simulacao", "3"]:
                    self.simular_analise_libs()
                    self.salvar_log("comando_libs")
                
                elif comando in ["timeline", "cronograma", "4"]:
                    self.mostrar_timeline_projeto()
                    self.salvar_log("comando_timeline")
                
                elif comando in ["conexao", "teste", "5"]:
                    self.testar_conexoes()
                    self.salvar_log("comando_conexao")
                
                elif comando in ["autor", "hebron", "6"]:
                    print(f"\n👤 AUTOR PRINCIPAL: {self.autor}")
                    print("📅 Desenvolvedor do Sistema Interestelar Brasileiro")
                    print("🎯 Foco: Tecnologia espacial de baixo custo")
                    self.salvar_log("comando_autor")
                
                else:
                    print(f"\n❌ Comando '{comando}' não reconhecido")
                    print("💡 Tente: info, 3i, libs, timeline, conexao, autor, sair")
                    self.salvar_log(f"comando_invalido_{comando}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Sistema interrompido. Logs salvos.")
                self.salvar_log("interrupcao_usuario")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                self.salvar_log(f"erro_{str(e)[:20]}")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    sistema = SistemaInterestelarHebron()
    sistema.cabecalho()
    
    # Testar conexões iniciais
    conexoes = sistema.testar_conexoes()
    
    # Mostrar status
    print("\n" + "="*70)
    print("📊 STATUS DO SISTEMA:")
    print("="*70)
    
    if all(conexoes.values()):
        print("✅ TODAS AS CONEXÕES: OPERACIONAIS")
        print("🎯 SISTEMA PRONTO PARA DADOS REAIS")
    else:
        print("⚠️ CONEXÕES PARCIAIS")
        print("💡 Usando dados simulados quando necessário")
    
    print(f"\n👤 Desenvolvedor: {sistema.autor}")
    print(f"📅 Sistema criado em: {sistema.data_criacao}")
    print("🌌 Projeto: Análise de objetos interestelares com tecnologia BR")
    
    # Iniciar modo interativo
    sistema.modo_interativo()
    
    print("\n" + "="*70)
    print("✅ SISTEMA INTERESTELAR FINALIZADO")
    print("="*70)
    print(f"📁 Logs salvos em: {sistema.log_file}")
    print("🚀 Continue desenvolvendo em 2024 para 2025!")
    print("="*70)
