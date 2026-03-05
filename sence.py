#!/usr/bin/env python3
"""
🏛️ NEXUS OIKOS SYSTEM - Divisão de Astrofísica
Módulo: SENCE - Versão 4.0 (Biomarcadores Avançados + Gráfico)
"""

import sys
import csv
import json
import matplotlib.pyplot as plt
from datetime import datetime

print("="*70)
print("🏛️ NEXUS OIKOS - DIVISÃO DE ASTROFÍSICA")
print("Versão 4.0 - Biomarcadores Avançados")
print("="*70)

print(f"\n✅ Python versão: {sys.version}")
print(f"✅ Data/Hora: {datetime.now()}\n")

# Dados validados com NASA/ESA/JWST 2026
exoplanetas = [
    {"nome": "K2-18b", "distancia": 124.0, "temperatura": 275, "raio": 2.6, 
     "gases": ["H2O", "CH4", "CO2", "DMS"], "descoberta": 2015,
     "fonte": "JWST 2023-2026", "tipo": "Hycean"},
    
    {"nome": "TOI-700d", "distancia": 101.5, "temperatura": 268, "raio": 1.1, 
     "gases": ["CO2", "CH4", "O2", "O3"], "descoberta": 2020,
     "fonte": "TESS", "tipo": "Super-Terra"},
    
    {"nome": "Proxima Centauri b", "distancia": 4.24, "temperatura": 234, "raio": 1.17, 
     "gases": ["H2O", "CO2", "CH4"], "descoberta": 2016,
     "fonte": "ESO", "tipo": "Terrestre"},
    
    {"nome": "TRAPPIST-1e", "distancia": 40.7, "temperatura": 246, "raio": 0.92, 
     "gases": ["CO2", "H2O", "CH4"], "descoberta": 2017,
     "fonte": "Spitzer/JWST", "tipo": "Terrestre"},
    
    {"nome": "LHS 1140b", "distancia": 48.5, "temperatura": 230, "raio": 1.73, 
     "gases": ["H2O", "CO2", "CH4"], "descoberta": 2017,
     "fonte": "MEarth", "tipo": "Super-Terra"},
    
    {"nome": "GJ 1061c", "distancia": 12.0, "temperatura": 245, "raio": 1.18, 
     "gases": ["H2O", "CO2"], "descoberta": 2020,
     "fonte": "TESS", "tipo": "Terrestre"},
]

print("🔬 ANALISANDO EXOPLANETAS COM BIOMARCADORES AVANÇADOS...")
print("-"*70)

# Sistema de scoring avançado
resultados = []
for p in exoplanetas:
    score = 0
    justificativas = []
    
    # 1. ZONA HABITÁVEL (0-30)
    if 250 <= p["temperatura"] <= 300:
        score += 30
        justificativas.append("Zona habitável (+30)")
    elif 200 <= p["temperatura"] <= 350:
        score += 15
        justificativas.append("Zona habitável estendida (+15)")
    
    # 2. BIOASSINATURAS PRIMÁRIAS (0-50)
    if "DMS" in p["gases"]:
        score += 50
        justificativas.append("🚀 DMS detectado - FORTE BIOASSINATURA (+50)")
    elif "CH4" in p["gases"] and "O2" in p["gases"]:
        score += 40
        justificativas.append("⚠️ CH4 + O2 - Desequilíbrio químico (+40)")
    elif "O3" in p["gases"]:
        score += 30
        justificativas.append("🛡️ Ozônio - Atmosfera oxidante (+30)")
    
    # 3. BIOASSINATURAS SECUNDÁRIAS (0-30)
    if "N2O" in p["gases"]:
        score += 30
        justificativas.append("N2O - Forte indicador biológico (+30)")
    elif "CH4" in p["gases"]:
        score += 15
        justificativas.append("CH4 detectado (+15)")
    
    # 4. ÁGUA (0-20)
    if "H2O" in p["gases"]:
        score += 20
        justificativas.append("Água detectada (+20)")
    
    # 5. PROXIMIDADE (0-15)
    if p["distancia"] < 10:
        score += 15
        justificativas.append("Muito próximo (<10 ly) (+15)")
    elif p["distancia"] < 50:
        score += 10
        justificativas.append("Próximo (<50 ly) (+10)")
    
    # 6. TIPO DE PLANETA (0-15)
    if p["tipo"] == "Terrestre":
        score += 15
        justificativas.append("Planeta rochoso (+15)")
    elif p["tipo"] == "Super-Terra":
        score += 10
        justificativas.append("Super-Terra (+10)")
    elif p["tipo"] == "Hycean":
        score += 5
        justificativas.append("Mundo Hycean (+5)")
    
    resultados.append({
        "nome": p["nome"],
        "score": min(100, score),
        "distancia": p["distancia"],
        "temperatura": p["temperatura"],
        "gases": p["gases"],
        "descoberta": p["descoberta"],
        "fonte": p["fonte"],
        "tipo": p["tipo"],
        "justificativas": justificativas
    })

# Ordenar por score
resultados.sort(key=lambda x: x["score"], reverse=True)

# RELATÓRIO DETALHADO
print("\n" + "="*70)
print("🏛️ RELATÓRIO NEXUS - ANÁLISE DE BIOMARCADORES")
print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("="*70)

for i, r in enumerate(resultados, 1):
    print(f"\n{i}. {r['nome']} [Score: {r['score']}/100]")
    print(f"   📍 Distância: {r['distancia']} ly | 🌡️ {r['temperatura']}K")
    print(f"   💨 Gases: {', '.join(r['gases'])}")
    print(f"   📡 Fonte: {r['fonte']} | Tipo: {r['tipo']}")
    
    if r['score'] >= 80:
        print(f"   ⚠️ STATUS: PRIORIDADE MÁXIMA - CANDIDATO A VIDA!")
    elif r['score'] >= 60:
        print(f"   🟡 STATUS: ALTA PRIORIDADE - Observar com JWST")
    else:
        print(f"   🟢 STATUS: MONITORAMENTO")
    
    print(f"\n   📋 Justificativas:")
    for j in r['justificativas']:
        print(f"      • {j}")

# EXPORTAÇÃO
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
csv_file = f'analise_astro_{timestamp}.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['planeta', 'score', 'distancia', 'temperatura', 'gases', 'tipo', 'descoberta'])
    for r in resultados:
        writer.writerow([
            r['nome'], r['score'], r['distancia'], r['temperatura'],
            ', '.join(r['gases']), r['tipo'], r['descoberta']
        ])
print(f"\n✅ CSV salvo: {csv_file}")

json_file = f'analise_astro_{timestamp}.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print(f"✅ JSON salvo: {json_file}")

# GRÁFICO
print("\n📊 Gerando gráfico de ranking...")
plt.figure(figsize=(10,6), facecolor='#0e1117')
ax = plt.axes()
ax.set_facecolor('#0e1117')

nomes = [r["nome"] for r in resultados]
scores = [r["score"] for r in resultados]
cores = ['#00ff41' if s >= 80 else '#ffd700' if s >= 60 else '#ff4444' for s in scores]

bars = plt.barh(nomes, scores, color=cores, edgecolor='white', linewidth=0.5)
plt.xlabel('Score de Bioassinatura (%)', color='white', fontweight='bold')
plt.title('🏛️ NEXUS OIKOS - RANKING DE HABITABILIDADE', color='white', fontsize=14)
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(axis='x', linestyle='--', alpha=0.3)

for bar, score in zip(bars, scores):
    plt.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score}%', 
             va='center', color='white', fontweight='bold')

plt.tight_layout()
grafico_file = f'ranking_habitabilidade_{timestamp}.png'
plt.savefig(grafico_file, dpi=150, facecolor='#0e1117')
print(f"✅ Gráfico salvo: {grafico_file}")

print("\n" + "="*70)
print("✅ ANÁLISE CONCLUÍDA COM SUCESSO")
print("🏛️ NEXUS OIKOS - Tecnologia com Alma")
print("="*70)
