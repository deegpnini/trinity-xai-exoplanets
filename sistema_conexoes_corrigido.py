#!/usr/bin/env python3
"""
SISTEMA COM CONEXÕES CORRIGIDAS
Autor: Helyton R. G. Ronchi (Hebron)
"""

import requests
from datetime import datetime

print("="*60)
print("🔧 CONEXÕES CORRIGIDAS - SISTEMA INTERESTELAR")
print("="*60)
print(f"👤 Autor: Helyton R. G. Ronchi (Hebron)")
print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("="*60)

def testar_conexoes_corrigidas():
    """Testa conexões com APIs que FUNCIONAM"""
    print("\n📡 TESTANDO CONEXÕES CORRIGIDAS...")
    print("-"*40)
    
    # APIs que FUNCIONAM sem chave especial
    conexoes_funcionais = {
        "Internet Geral": "https://www.google.com",
        "NASA API (Demo)": "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",
        "Space-Track (Info)": "https://www.space-track.org",
        "ESA Website": "https://www.esa.int",
        "JPL Horizons (Info)": "https://ssd.jpl.nasa.gov",
        "GitHub API": "https://api.github.com",
        "Open APIs BR": "https://brasilapi.com.br/api/cep/v1/01001000"
    }
    
    resultados = {}
    for nome, url in conexoes_funcionais.items():
        try:
            if "DEMO_KEY" in url:
                # NASA Demo funciona
                resp = requests.get(url, timeout=5)
            else:
                resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            
            if resp.status_code == 200:
                resultados[nome] = "✅ CONECTADO"
                print(f"  {nome}: ✅ CONECTADO")
            elif resp.status_code == 403:
                resultados[nome] = "⚠️ ACESSO NEGADO (normal)"
                print(f"  {nome}: ⚠️ ACESSO NEGADO (normal para APIs restritas)")
            else:
                resultados[nome] = f"⚠️ Código {resp.status_code}"
                print(f"  {nome}: ⚠️ Código {resp.status_code}")
                
        except requests.exceptions.Timeout:
            resultados[nome] = "⏰ TIMEOUT"
            print(f"  {nome}: ⏰ TIMEOUT")
        except Exception as e:
            resultados[nome] = f"❌ {str(e)[:30]}"
            print(f"  {nome}: ❌ {str(e)[:30]}")
    
    return resultados

def mostrar_apis_recomendadas():
    """Mostra APIs que realmente funcionam para o projeto"""
    print("\n🎯 APIs RECOMENDADAS PARA SEU PROJETO:")
    print("-"*40)
    
    apis = {
        "NASA APOD": {
            "url": "https://api.nasa.gov/planetary/apod",
            "chave": "DEMO_KEY (grátis)",
            "uso": "Imagens astronômicas diárias"
        },
        "NASA NEO": {
            "url": "https://api.nasa.gov/neo/rest/v1",
            "chave": "DEMO_KEY (limitado)",
            "uso": "Dados de objetos próximos da Terra"
        },
        "OpenNotify ISS": {
            "url": "http://api.open-notify.org/iss-now.json",
            "chave": "Nenhuma",
            "uso": "Posição da Estação Espacial em tempo real"
        },
        "Brazil API": {
            "url": "https://brasilapi.com.br",
            "chave": "Nenhuma",
            "uso": "APIs públicas brasileiras (exemplo)"
        },
        "GitHub API": {
            "url": "https://api.github.com",
            "chave": "Opcional para limites maiores",
            "uso": "Armazenar código do projeto"
        }
    }
    
    for nome, info in apis.items():
        print(f"\n📡 {nome}:")
        print(f"   URL: {info['url']}")
        print(f"   Chave: {info['chave']}")
        print(f"   Uso: {info['uso']}")

def explicar_minor_planet_error():
    """Explica o erro do Minor Planet Center"""
    print("\n🔍 ENTENDA O ERRO DO MINOR PLANET CENTER:")
    print("-"*40)
    print("""
    ❓ POR QUE DEU ERRO?
    
    O Minor Planet Center (MPC) é o órgão OFICIAL que:
    • Mantém o catálogo de TODOS asteroides e cometas
    • Atribui designações oficiais (ex: 3I/ATLAS)
    • Valida descobertas científicas
    
    ⚠️ RESTRIÇÕES DE ACESSO:
    1. API NÃO é pública
    2. Acesso apenas para observatórios credenciados
    3. Dados brutos só para pesquisadores autorizados
    4. Bloqueia requisições automáticas não autorizadas
    
    ✅ SOLUÇÕES PARA VOCÊ:
    
    1. USAR APIs PÚBLICAS ALTERNATIVAS:
       • NASA NEO API (DEMO_KEY funciona)
       • JPL Horizons (para efemérides)
       • Space-Track (para dados orbitais)
    
    2. ACESSAR DADOS VIA WEB (manual):
       • https://minorplanetcenter.net (site oficial)
       • Baixar arquivos CSV públicos
       • Usar dados de observatórios parceiros
    
    3. PARA SEU PROJETO 3I/ATLAS:
       • Quando 3I/ATLAS for confirmado (2025)
       • Os dados serão públicos no MPC
       • Você poderá acessar via site ou APIs secundárias
    """)

# Executar
print("\n" + "="*60)
print("🚀 INICIANDO TESTES CORRIGIDOS...")
print("="*60)

resultados = testar_conexoes_corrigidas()

print("\n" + "="*60)
print("📊 RESULTADOS FINAIS:")
print("="*60)

conexoes_ok = sum(1 for r in resultados.values() if "✅" in r)
total_conexoes = len(resultados)

print(f"✅ Conexões bem-sucedidas: {conexoes_ok}/{total_conexoes}")

if conexoes_ok >= 4:
    print("🎯 SISTEMA OPERACIONAL PARA SEU PROJETO!")
    print("💡 Você tem conexão suficiente para:")
    print("   • Acessar dados da NASA")
    print("   • Consultar APIs públicas")
    print("   • Desenvolver sistema interestelar")
else:
    print("⚠️ CONEXÃO LIMITADA")
    print("💡 Sugestão: Use dados simulados e APIs locais")

explicar_minor_planet_error()
mostrar_apis_recomendadas()

print("\n" + "="*60)
print("🎯 PRÓXIMOS PASSOS PARA HEBRON:")
print("="*60)
print("1. Focar nas APIs que FUNCIONAM (NASA Demo)")
print("2. Criar sistema com dados simulados + reais")
print("3. Quando 3I/ATLAS for descoberto (2025)")
print("   • Dados serão públicos")
print("   • MPC liberará informações")
print("   • Seu sistema estará pronto!")
print("\n💡 Lembre: Desenvolvimento AGORA (2024)")
print("          Dados reais DEPOIS (2025)")

print("\n" + "="*60)
print("✅ Sistema de conexões analisado!")
print("="*60)
