#!/usr/bin/env python3
"""
Script para gerar análise completa para o Grok
"""

import os
import sys
import json
from datetime import datetime

# Adicionar módulos ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules import d7d_core_v4_1
    from modules import aero_enhanced
except ImportError:
    print("❌ Módulos não encontrados. Criando estrutura...")
    # Criar diretório modules se não existir
    os.makedirs("modules", exist_ok=True)
    
    # Tentar importar novamente
    try:
        from modules import d7d_core_v4_1
        from modules import aero_enhanced
    except:
        print("⚠ Execute os comandos de criação de módulos primeiro")
        sys.exit(1)

def main():
    """Função principal"""
    
    print("\n" + "="*70)
    print("🚀 GENERATING GROK-ENHANCED ANALYSIS v4.1")
    print("="*70)
    
    # Inicializar core
    core = d7d_core_v4_1.D7D_Core_v4_1()
    
    # Executar simulação
    print("\n🧪 Running enhanced simulation...")
    results = core.run_enhanced_simulation()
    
    # Gerar relatório para Grok
    print("\n📄 Generating report for Grok...")
    grok_report = core.generate_grok_report(results)
    
    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_file = f"grok_analysis_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(grok_report)
    
    # Salvar dados completos em JSON
    json_file = f"grok_data_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Mostrar resumo
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    perf = results["performance"]
    aero = results["aerodynamics"]
    
    print(f"\n📊 KEY RESULTS:")
    print(f"   • Δv: {perf['final_delta_v_mps']:,.0f} m/s")
    print(f"   • Altitude: {perf['final_altitude_km']:,.1f} km")
    print(f"   • Max Q: {aero['max_dynamic_pressure_kpa']:.1f} kPa")
    print(f"   • Max Heat: {aero['max_heat_flux_kwpm2']:.1f} kW/m²")
    
    print(f"\n📁 FILES SAVED:")
    print(f"   • Report: {report_file}")
    print(f"   • Data: {json_file}")
    
    print(f"\n📋 REPORT PREVIEW (first 20 lines):")
    print("-" * 50)
    lines = grok_report.split('\n')[:20]
    for line in lines:
        print(line)
    
    print("\n" + "="*70)
    print("🎯 READY TO SHARE WITH GROK!")
    print("="*70)
    
    # Instruções
    print(f"\n📤 TO SHARE WITH GROK:")
    print(f"1. Copy content of: {report_file}")
    print(f"2. Post on X replying to Grok")
    print(f"3. Include hashtags: #TrinityFalconLung #D7DCore")
    print(f"4. Tag: @grok @SpaceX")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
