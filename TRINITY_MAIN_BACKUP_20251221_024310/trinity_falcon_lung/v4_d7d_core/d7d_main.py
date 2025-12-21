#!/usr/bin/env python3
"""
D7D_CORE MAIN EXECUTOR
Trinity Falcon Lung v4.0 - D7D Quality
"""

import sys
import time
from modules import d7d_core

def main():
    print("🚀" * 35)
    print("       TRINITY FALCON LUNG v4.0 - D7D_CORE QUALITY")
    print("🚀" * 35)
    
    # Inicializar D7D Core
    engine = d7d_core.D7D_Core_Engine()
    
    # Executar simulação
    results = engine.run_complete_simulation(max_time=162)
    
    # Gerar relatório
    report_file = engine.generate_report(results)
    
    # Mostrar insights D7D
    print("\n" + "💡" * 35)
    print("       D7D_CORE INSIGHTS")
    print("💡" * 35)
    
    # Calcular payload extra
    fuel_saving = results["performance"]["fuel_saving_percent"]
    extra_payload_kg = 4000 * (fuel_saving / 100)  # Base: 4000kg payload
    
    # Valor econômico
    cost_per_kg = 1500  # R$/kg estimado
    savings_per_launch = extra_payload_kg * cost_per_kg
    
    print(f"\n💰 ECONOMIC IMPACT D7D:")
    print(f"   • Extra Payload Capacity: +{extra_payload_kg:.0f} kg")
    print(f"   • Value per Launch: R$ {savings_per_launch:,.0f}")
    print(f"   • Annual Savings (10 launches): R$ {savings_per_launch * 10:,.0f}")
    
    print(f"\n🌍 ENVIRONMENTAL IMPACT:")
    print(f"   • Fuel Saved: {fuel_saving:.2f}% per launch")
    print(f"   • CO₂ Reduction: {(fuel_saving * 300):.0f} tons/year")
    print(f"   • Energy Recovery: {results['d7d_enhancements']['energy_recovery_potential_kwh']:.1f} kWh")
    
    print(f"\n🔬 SCIENTIFIC CONTRIBUTION:")
    print(f"   • Data Points Collected: {results['d7d_enhancements']['telemetry_data_points']:,}")
    print(f"   • Optimization Iterations: {results['d7d_enhancements']['ai_optimization_iterations']}")
    print(f"   • Layers Activated: {results['d7d_enhancements']['total_layers_active']}/7")
    
    print("\n" + "🚀" * 35)
    print("       D7D_CORE MISSION ACCOMPLISHED")
    print("🚀" * 35)
    
    print(f"\n📁 Output files:")
    print(f"   • Report: {report_file}")
    print(f"   • Telemetry: logs/")
    print(f"   • Configuration: config/")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ D7D_CORE ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
