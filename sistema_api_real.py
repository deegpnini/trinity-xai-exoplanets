#!/usr/bin/env python3
"""
SISTEMA COM DADOS REAIS DA NASA/ESA
Autor: Helyton R. G. Ronchi (Hebron)
"""

import requests
from datetime import datetime

print("="*60)
print("🚀 SISTEMA COM DADOS REAIS DA NASA/ESA")
print("="*60)
print("👤 Autor: Helyton R. G. Ronchi (Hebron)")
print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("="*60)

def testar_conexao_nasa():
    """Testa conexão com API da NASA"""
    print("\n📡 TESTANDO CONEXÃO COM NASA...")
    try:
        # API pública da NASA para NEOs
        url = "https://api.nasa.gov/neo/rest/v1/neo/browse?api_key=DEMO_KEY"
        resposta = requests.get(url, timeout=5)
        
        if resposta.status_code == 200:
            dados = resposta.json()
            total_neos = dados["page"]["total_elements"]
            print(f"✅ NASA: CONECTADO")
            print(f"📊 NEOs catalogados: {total_neos:,}")
            return True
        else:
            print(f"⚠️ NASA: Erro {resposta.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ NASA: Falha na conexão - {str(e)[:50]}")
        return False

def testar_conexao_esa():
    """Testa conexão com ESA"""
    print("\n🌍 TESTANDO CONEXÃO COM ESA...")
    try:
        # Site da ESA
        url = "https://www.esa.int"
        resposta = requests.get(url, timeout=5)
        
        if resposta.status_code == 200:
            print(f"✅ ESA: CONECTADO")
            print(f"📊 Status: Site acessível")
            return True
        else:
            print(f"⚠️ ESA: Erro {resposta.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ESA: Falha na conexão - {str(e)[:50]}")
        return False

def mostrar_dados_simulados_3i():
    """Mostra dados simulados do 3I/ATLAS"""
    print("\n📊 DADOS SIMULADOS - 3I/ATLAS:")
    print("-"*40)
    
    dados_3i = {
        "designacao": "3I/ATLAS",
        "descoberta": "Julho 2025",
        "perihelio": "Dezembro 2025",
        "distancia_minima": "0.05 UA",
        "magnitude": "~18",
        "tipo": "Objeto interestelar",
        "observatorios": ["ATLAS-Hawaii", "Pan-STARRS", "Catalina"]
    }
    
    for chave, valor in dados_3i.items():
        print(f"• {chave.replace('_', ' ').title()}: {valor}")
    
    print("\n💡 Estes são dados SIMULADOS baseados em:")
    print("   - Padrões de objetos interestelares anteriores")
    print("   - Capacidades do sistema ATLAS")
    print("   - Previsões astronômicas para 2025")

# Executar testes
print("\n" + "="*60)
print("🔍 INICIANDO TESTES DE CONEXÃO...")
print("="*60)

nasa_ok = testar_conexao_nasa()
esa_ok = testar_conexao_esa()

print("\n" + "="*60)
print("📈 RESULTADO DOS TESTES:")
print("="*60)

if nasa_ok and esa_ok:
    print("✅ SISTEMA PRONTO PARA DADOS REAIS!")
    print("   • NASA: Conectada")
    print("   • ESA: Conectada")
    print("   • Próximo passo: Integrar API real do Minor Planet Center")
elif nasa_ok or esa_ok:
    print("⚠️ CONEXÃO PARCIAL")
    if nasa_ok:
        print("   • NASA: Conectada ✓")
        print("   • ESA: Falhou ✗")
    else:
        print("   • NASA: Falhou ✗")
        print("   • ESA: Conectada ✓")
else:
    print("❌ SEM CONEXÃO COM FONTES OFICIAIS")
    print("   • Usando dados simulados")

# Mostrar dados do 3I/ATLAS (simulados por enquanto)
mostrar_dados_simulados_3i()

print("\n" + "="*60)
print("🎯 PRÓXIMOS PASSOS PARA HEBRON:")
print("="*60)
print("1. Obter API Key gratuita da NASA")
print("2. Acessar dados do Minor Planet Center")
print("3. Integrar com sistema de alerta de objetos")
print("4. Criar simulações com dados reais")
print("\n💡 Dica: NASA API Key gratuita em:")
print("   https://api.nasa.gov")

print("\n" + "="*60)
print("✅ Sistema de verificação concluído!")
print("="*60)
