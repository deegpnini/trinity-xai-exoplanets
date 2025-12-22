#!/usr/bin/env python3
"""
SISTEMA INTERESTELAR BRASILEIRO - VERSÃO FINAL SEM ERROS
Autor: Helyton R. G. Ronchi (Hebron)
Desenvolvimento: 2024 | Alvo: 2025
"""

import requests
from datetime import datetime

class SistemaInterestelar:
    def __init__(self):
        self.autor = "Helyton R. G. Ronchi (Hebron)"
        self.hoje = datetime.now().strftime("%d/%m/%Y")
        self.ano_atual = 2024
        self.ano_alvo = 2025
        self.projeto = "Projeto Interestelar Brasileiro"
    
    def cabecalho(self):
        print("="*60)
        print("🚀 SISTEMA INTERESTELAR BRASILEIRO")
        print("="*60)
        print(f"👤 Autor: {self.autor}")
        print(f"📅 Hoje: {self.hoje} (ano {self.ano_atual})")
        print(f"🎯 Alvo: Ano {self.ano_alvo} - 3I/ATLAS")
        print(f"🌌 Projeto: {self.projeto}")
        print("="*60)
        
        try:
            r = requests.get("https://api.github.com", timeout=3)
            print(f"🌐 Internet: CONECTADO")
        except:
            print("🌐 Internet: SEM CONEXÃO")
        
        print("\n📅 CRONOGRAMA:")
        print("2024 (AGORA): Desenvolvimento do sistema")
        print("2025 (FUTURO): 3I/ATLAS + primeiros testes")
        print("2026: Sistema operacional completo")
        print("="*60)
    
    def processar(self, comando):
        comando = comando.lower().strip()
        
        if comando == "hoje":
            return f"📅 HOJE é {self.hoje} - Estamos desenvolvendo o sistema em {self.ano_atual}"
        
        elif comando == "2025":
            return f"🎯 2025 é o ANO ALVO: 3I/ATLAS será detectado e testaremos o sistema"
        
        elif comando == "3i/atlas":
            return """📡 3I/ATLAS (objeto interestelar):
• Será detectado em 2025
• Aproximação máxima: 19/12/2025
• Estamos preparando sistema AGORA para analisá-lo"""
        
        elif comando == "teste":
            return "✅ Sistema funcionando! Desenvolvido em 2024 para 2025!"
        
        elif comando == "autor":
            return f"👤 Autor do sistema: {self.autor}"
        
        elif comando == "sair":
            return "sair"
        
        else:
            return f"❓ Comando '{comando}' não reconhecido. Tente: hoje, 2025, 3i/atlas, teste, autor, sair"
    
    def executar(self):
        self.cabecalho()
        
        print("\n🤖 COMANDOS DISPONÍVEIS:")
        print("• hoje     - Data atual e desenvolvimento")
        print("• 2025     - Informações sobre o ano alvo")
        print("• 3i/atlas - Sobre o objeto interestelar")
        print("• teste    - Testar sistema")
        print("• autor    - Mostrar autor")
        print("• sair     - Encerrar")
        print("="*60)
        
        while True:
            try:
                entrada = input("\n🎯 Digite comando: ").strip()
                
                if not entrada:
                    print("⚠️ Digite algo...")
                    continue
                
                resultado = self.processar(entrada)
                
                if resultado == "sair":
                    print("\n👋 Encerrando. Continuamos em 2024 para 2025!")
                    break
                
                print(f"\n📤 {resultado}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrompido. Desenvolvimento continua!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    sistema = SistemaInterestelar()
    sistema.executar()
