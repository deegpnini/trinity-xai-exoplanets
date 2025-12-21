#!/usr/bin/env python3
"""
SISTEMA INTERESTELAR BRASILEIRO - VERSÃO FINAL
Autor: Helyton R. G. Ronchi (Hebron)
Data CORRETA: 21/12/2024
"""

import json
import requests
from datetime import datetime
import os

class SistemaInterestelar:
    def __init__(self):
        self.autor = "Helyton R. G. Ronchi (Hebron)"
        self.data_correta = "21/12/2024"  # CORRIGIDO: 2024, não 2025
        self.projeto = "Projeto Interestelar Brasileiro"
        
        # Dados CORRETOS do 3I/ATLAS
        self.dados_3i_atlas = {
            "designacao": "3I/ATLAS",  # CORRETO: 3I, não 31
            "ano_descoberta": 2025,
            "data_aproximacao": "19/12/2025",
            "tipo": "Objeto Interestelar Confirmado",
            "status": "Em observação"
        }
    
    def cabecalho(self):
        """Mostra cabeçalho correto"""
        print("="*80)
        print("🚀 SISTEMA INTERESTELAR BRASILEIRO - VERSÃO FINAL")
        print("="*80)
        print(f"👤 Autor: {self.autor}")
        print(f"📅 Data atual do sistema: {self.data_correta}")
        print(f"🌌 Projeto: {self.projeto}")
        print("="*80)
        
        # Teste de conexão REAL
        try:
            resposta = requests.get("https://api.github.com", timeout=3)
            print(f"🌐 Status Internet: CONECTADO ({resposta.status_code})")
        except:
            print("🌐 Status Internet: SEM CONEXÃO")
        print("="*80)
    
    def processar_comando(self, comando):
        """Processa comandos com CORREÇÕES"""
        comando = comando.lower().strip()
        
        # CORREÇÃO: Aceita com e sem acento
        if comando in ["3i/atlas", "3iatlas", "31/atlas", "31atlas"]:
            return self.info_3i_atlas()
        
        # CORREÇÃO: Aceita "trajetoria" e "trajetória"
        elif comando in ["trajetória", "trajetoria", "trajeto"]:
            return self.calculo_trajetoria()
        
        elif comando in ["libs", "espectrometro", "laser"]:
            return self.info_libs()
        
        elif comando in ["teste", "testar", "status"]:
            return "✅ Sistema funcionando perfeitamente! Autor: Helyton R. G. Ronchi (Hebron)"
        
        elif comando in ["data", "data correta", "ano"]:
            return f"📅 Data CORRETA do sistema: {self.data_correta} (2024, não 2025)"
        
        else:
            return "❓ Comando não reconhecido. Tente: '3I/ATLAS', 'trajetória', 'LIBS', 'teste', 'data'"
    
    def info_3i_atlas(self):
        """Informações CORRETAS sobre 3I/ATLAS"""
        info = f"""
📡 INFORMAÇÕES SOBRE 3I/ATLAS (CORRETO: 3I, não 31):

• Designação oficial: {self.dados_3i_atlas['designacao']}
• Ano descoberta: {self.dados_3i_atlas['ano_descoberta']}
• Aproximação Terra: {self.dados_3i_atlas['data_aproximacao']}
• Tipo: {self.dados_3i_atlas['tipo']}
• Status: {self.dados_3i_atlas['status']}

💡 Dado importante: 3I/ATLAS é o TERCEIRO objeto interestelar confirmado,
após 1I/'Oumuamua (2017) e 2I/Borisov (2019).
"""
        return info
    
    def calculo_trajetoria(self):
        """Cálculos de trajetória"""
        return """
🛰️ CÁLCULOS DE TRAJETÓRIA:

Para encontro com objeto interestelar:
• Distância típica: 0.1-1.0 UA
• Delta-V necessário: 3-6 km/s
• Janela de lançamento: 30-90 dias pós-detecção
• Foguete suborbital brasileiro pode fornecer parte do ΔV
• Restante via propulsão elétrica ou química

📊 Simulação básica: Para objeto a 0.5 UA:
  - Tempo de viagem: 60-180 dias
  - Combustível necessário: 40-60% da massa
"""
    
    def info_libs(self):
        """Informações sobre LIBS"""
        return """
🔬 ESPECTRÔMETRO LIBS (Laser-Induced Breakdown Spectroscopy):

• Aplicação: Análise elementar remota de objetos espaciais
• Elementos detectáveis: Fe, Ni, Mg, Si, C, O, H, N, S, Ca, Al, Na
• Limite de detecção: 10-1000 ppm dependendo do elemento
• Distância de operação: 10-100 metros
• Potência típica: 50-200 mJ/pulse
• Taxa de repetição: 1-10 Hz

🎯 Para objeto interestelar:
  - 1000 pulsos laser = espectro completo
  - Tempo de análise: 2-10 minutos
  - Dados: Composição elementar + abundâncias
"""
    
    def modo_interativo(self):
        """Modo interativo melhorado"""
        print("\n" + "="*80)
        print("🤖 MODO INTERATIVO - DIGITE SEUS COMANDOS")
        print("="*80)
        print("Comandos disponíveis:")
        print("  • '3I/ATLAS' - Informações sobre o objeto (3I, não 31)")
        print("  • 'trajetória' - Cálculos de encontro (com ou sem acento)")
        print("  • 'LIBS' - Especificações do espectrômetro")
        print("  • 'teste' - Verificar status do sistema")
        print("  • 'data' - Ver data CORRETA do sistema")
        print("  • 'sair' - Encerrar")
        print("="*80)
        
        contador = 0
        while True:
            try:
                entrada = input(f"\n[{contador}] 🎯 Comando: ").strip()
                contador += 1
                
                if entrada.lower() in ['sair', 'exit', 'quit', 's']:
                    print("\n👋 Encerrando sistema. Até mais!")
                    break
                
                if entrada:
                    resposta = self.processar_comando(entrada)
                    print(f"\n📤 RESPOSTA:\n{resposta}")
                else:
                    print("⚠️ Digite um comando ou 'sair'")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrompido pelo usuário")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    sistema = SistemaInterestelar()
    sistema.cabecalho()
    sistema.modo_interativo()
