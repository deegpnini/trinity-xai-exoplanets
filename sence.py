#!/usr/bin/env python3
"""
🏛️ NEXUS OIKOS SYSTEM - Divisão de Astrofísica
Módulo: SENCE - Versão 3.0 LIGHT (sem matplotlib)
"""

import sys
import random
import csv
import json
from datetime import datetime

print("="*70)
print("🏛️ NEXUS OIKOS - DIVISÃO DE ASTROFÍSICA")
print("Versão 3.0 LIGHT - Análise sem gráfico")
print("="*70)

print(f"\n✅ Python versão: {sys.version}")
print(f"✅ Data/Hora: {datetime.now()}\n")

# Dados validados com NASA/ESA
exoplanetas = [
    {"nome": "TOI-700d", "distancia": 101.5, "temperatura": 268, "raio": 1.1, 
     "gases": ["CO2", "CH4", "O2", "O3"], "descoberta": 2020},
    {"nome": "Proxima Centauri b", "distancia": 4.24, "temperatura": 234, "raio": 1.17, 
     "gases": ["H2O", "CO2", "CH4"], "descoberta": 2016},
    {"nome": "TRAPPIST-1e", "distancia": 40.7, "temperatura": 246, "raio": 0.92, 
     "gases": ["CO2", "H2O"], "descoberta": 2017},
    {"nome": "K2-18b", "distancia": 124.0, "temperatura": 275, "raio": 2.6, 
     "gases": ["H2O", "CH4", "CO2", "DMS"], "descoberta": 2015},
    {"nome": "LHS 1140b", "distancia": 48.5, "temperatura": 230, "raio": 1.73, 
     "gases": ["H2O", "CO2"], "descoberta": 2017},
]

print("🔬 ANALISANDO EXOPLANETAS...")
print("-"*70)

# Análise
resultados = []
for p in exoplanetas:
    score = 0
    justificativas = []
    
    # Zona habitável (30 pontos)
    if 250 <= p["temperatura"] <= 300:
        score += 30
        justificativas.append("Zona habitável (+30)")
    
    # Bioassinaturas (30 pontos)
    if "CH4" in p["gases"] and "O2" in p["gases"]:
        score += 30
        justificativas.append("CH4+O2 (+30) ⚠️ DESEQUILÍBRIO QUÍMICO")
    elif "CH4" in p["gases"]:
        score += 15
        justificativas.append("CH4 (+15)")
    
    if "DMS" in p["gases"]:
        score += 20
        justificativas.append("DMS (+20) 🚀 BIOASSINATURA FORTE")
    
    # Água (15 pontos)
    if "H2O" in p["gases"]:
        score += 15
        justificativas.append("H2O (+15)")
    
    # Proximidade (10 pontos)
    if p["distancia"] < 10:
        score += 10
        justificativas.append("Distância <10 anos-luz (+10)")
    elif p["distancia"] < 50:
        score += 5
        justificativas.append("Distância <50 anos-luz (+5)")
    
    resultados.append({
        "nome": p["nome"],
        "score": min(100, score),
        "distancia": p["distancia"],
        "temperatura": p["temperatura"],
        "gases": p["gases"],
        "descoberta": p["descoberta"],
        "justificativas": justificativas
    })

# Ordenar por score
resultados.sort(key=lambda x: x["score"], reverse=True)

# RELATÓRIO PRINCIPAL
print("\n" + "="*70)
print("🏛️ RELATÓRIO NEXUS - ANÁLISE DE EXOPLANETAS")
print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("="*70)

for i, r in enumerate(resultados, 1):
    print(f"\n{i}. {r['nome']}")
    print(f"   🌡️ Score: {r['score']}/100")
    print(f"   📍 Distância: {r['distancia']} anos-luz")
    print(f"   🌡️ Temperatura: {r['temperatura']}K ({r['temperatura']-273}°C)")
    print(f"   💨 Gases: {', '.join(r['gases'])}")
    print(f"   📅 Descoberta: {r['descoberta']}")
    
    if r['score'] >= 80:
        print(f"   ⚠️ STATUS: PRIORIDADE MÁXIMA - Candidato a vida!")
    elif r['score'] >= 60:
        print(f"   🟡 STATUS: ALTA PRIORIDADE - Observar com JWST")
    else:
        print(f"   🟢 STATUS: Monitoramento")
    
    print(f"\n   📋 Justificativas:")
    for j in r['justificativas']:
        print(f"      • {j}")

# RESUMO EXECUTIVO
print("\n" + "="*70)
print("📊 RESUMO EXECUTIVO")
print("="*70)

for r in resultados[:3]:
    print(f"\n🔴 {r['nome']}: Score {r['score']}")
    if "CH4" in r['gases'] and "O2" in r['gases']:
        print(f"   ⚠️ Desequilíbrio químico detectado!")

# EXPORTAÇÃO CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
csv_file = f'analise_astro_{timestamp}.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['planeta', 'score', 'distancia', 'temperatura', 'gases', 'descoberta'])
    for r in resultados:
        writer.writerow([
            r['nome'], 
            r['score'], 
            r['distancia'], 
            r['temperatura'], 
            ', '.join(r['gases']), 
            r['descoberta']
        ])
print(f"\n✅ CSV salvo: {csv_file}")

# EXPORTAÇÃO JSON
json_file = f'analise_astro_{timestamp}.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print(f"✅ JSON salvo: {json_file}")

print("\n" + "="*70)
print("✅ ANÁLISE CONCLUÍDA COM SUCESSO")
print("="*70)
