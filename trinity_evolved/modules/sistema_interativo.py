#!/usr/bin/env python3
print("="*60)
print("📁 ARQUIVO 3: SISTEMA INTERATIVO")
print("="*60)
print("👤 Autor: Helyton R. G. Ronchi (Hebron)")
print("🔄 Modo: Interativo")
print("="*60)

contador = 1
while True:
    print(f"\n[{contador}] Digite algo (ou 'sair'):")
    entrada = input(">>> ").strip()
    
    if entrada.lower() == "sair":
        print("👋 Encerrando...")
        break
    
    print(f"📤 Você digitou: {entrada}")
    contador += 1

print("\n" + "="*60)
print("✅ Sistema finalizado!")
print("="*60)
