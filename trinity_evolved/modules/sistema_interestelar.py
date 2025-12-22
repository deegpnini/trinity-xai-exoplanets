#!/usr/bin/env python3
"""
SISTEMA INTERESTELAR BRASILEIRO
Autor: Helyton R. G. Ronchi (Hebron)
"""

print("="*60)
print("🚀 SISTEMA INTERESTELAR BRASILEIRO")
print("="*60)
print("👤 Autor: Helyton R. G. Ronchi (Hebron)")
print("📅 Data: 21/12/2024")
print("🎯 Alvo: 2025 - 3I/ATLAS")
print("="*60)

# Dados do projeto
dados = {
    "objetos_interstelares": [
        {"nome": "1I/'Oumuamua", "ano": 2017, "tipo": "asteroide?"},
        {"nome": "2I/Borisov", "ano": 2019, "tipo": "cometa"},
        {"nome": "3I/ATLAS", "ano": 2025, "tipo": "a confirmar"}
    ],
    "tecnologia": "LIBS (Laser-Induced Breakdown Spectroscopy)",
    "plataforma": "Foguete suborbital brasileiro",
    "meta": "Análise in situ do próximo objeto interestelar"
}

print("\n📊 DADOS DO PROJETO:")
print(f"• Tecnologia: {dados['tecnologia']}")
print(f"• Plataforma: {dados['plataforma']}")
print(f"• Meta: {dados['meta']}")

print("\n📡 OBJETOS INTERESTELARES CONHECIDOS:")
for obj in dados["objetos_interstelares"]:
    print(f"  • {obj['nome']} ({obj['ano']}) - {obj['tipo']}")

print("\n" + "="*60)
print("🤖 MODO INTERATIVO - DIGITE COMANDOS")
print("="*60)

comandos_validos = ["info", "3i/atlas", "tecnologia", "sair"]

while True:
    print("\nComandos disponíveis:", ", ".join(comandos_validos))
    cmd = input(">>> ").strip().lower()
    
    if cmd == "sair":
        print("\n👋 Até logo! Desenvolvimento continua em 2024!")
        break
    
    elif cmd == "info":
        print("\nℹ️  INFORMAÇÕES:")
        print(f"Autor: Helyton R. G. Ronchi (Hebron)")
        print(f"Projeto: Sistema para análise de objetos interestelares")
        print(f"Status: Em desenvolvimento (2024) para uso em 2025")
    
    elif cmd == "3i/atlas":
        print("\n📡 3I/ATLAS:")
        print("• 3º objeto interestelar confirmado")
        print("• Descoberta: 2025 (futuro próximo)")
        print("• Aproximação Terra: 19/12/2025")
        print("• Nosso alvo: Analisar com sistema LIBS")
    
    elif cmd == "tecnologia":
        print("\n🔬 TECNOLOGIA LIBS:")
        print("• Laser-Induced Breakdown Spectroscopy")
        print("• Detecta: Fe, Ni, Mg, Si, C, O, H, N, S")
        print("• Distância: 10-100 metros")
        print("• Aplicação: Análise elementar remota")
    
    else:
        print(f"\n❌ Comando '{cmd}' não reconhecido")
        print("Tente: info, 3i/atlas, tecnologia, sair")

print("\n" + "="*60)
print("✅ Sistema interestelar finalizado!")
print("="*60)
