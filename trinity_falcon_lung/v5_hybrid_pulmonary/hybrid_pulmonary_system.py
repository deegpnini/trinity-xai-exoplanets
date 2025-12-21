import time
import math
import json
import random
import os
import sys

# Configuração de Cores para o Terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TrinityHybridSystem:
    def __init__(self):
        self.gravity = 9.81
        self.fuel_mass = 50000  # kg
        self.dry_mass = 2500    # kg
        self.altitude = 0       # m
        self.velocity = 0       # m/s
        
        # Parâmetros do Sistema Híbrido
        self.battery_level = 100.0 # %
        self.eolic_energy_generated = 0.0 # kWh
        self.mode = "PRE-LAUNCH" 

    def run_simulation(self):
        os.system('clear')
        print(f"{Colors.HEADER}🚀 TRINITY FALCON LUNG v5.0 - INICIANDO SISTEMA...{Colors.ENDC}")
        print(f"{Colors.CYAN}ℹ️  Configuração: Híbrido (Combustão + Elétrico + Eólico){Colors.ENDC}")
        print("-" * 60)
        time.sleep(1.5)
        
        # FASE 1: SUBIDA (Combustão + Elétrico)
        print(f"{Colors.WARNING}🔥 FASE 1: ASCENSÃO (Modo Híbrido Ativo){Colors.ENDC}")
        print("   Injetando O2 Pulmonar Variável...")
        time.sleep(1)

        for t in range(0, 101, 20):
            # Física simplificada para demonstração
            alt_gain = (t * 120) + random.randint(0, 50)
            self.altitude += alt_gain
            fuel_burn = 400 - (t * 0.5) 
            self.fuel_mass -= fuel_burn
            efficiency = 100 + (t * 0.15)
            
            # Barra de progresso visual
            bar = "█" * int(t/10) + "░" * (10 - int(t/10))
            
            print(f"   ⏱️ T+{t}s [{bar}] Alt: {self.altitude}m | Combustível: {int(self.fuel_mass)}kg | Eficiência: {efficiency:.1f}%")
            time.sleep(0.3)
            
        # FASE 2: DESCIDA (Regeneração Eólica)
        print("-" * 60)
        print(f"{Colors.GREEN}♻️  FASE 2: DESCIDA & REGENERAÇÃO (Modo Eólico Ativo){Colors.ENDC}")
        print(f"   🌬️ Abrindo aletas de captação de fluxo...")
        time.sleep(1)
        
        descent_steps = 5
        energy_gain_base = 6.8
        
        for d in range(descent_steps):
            generated = energy_gain_base + random.uniform(0, 0.5)
            self.eolic_energy_generated += generated
            print(f"   ⚡ Turbinas girando... Capturado: +{generated:.2f} kWh | Total: {self.eolic_energy_generated:.2f} kWh")
            time.sleep(0.4)

        return self.generate_report()

    def generate_report(self):
        print("-" * 60)
        print(f"{Colors.BOLD}🏆 RELATÓRIO FINAL DE MISSÃO{Colors.ENDC}")
        print("-" * 60)
        
        # Cálculos finais
        savings = 350000.00
        co2_saved = self.eolic_energy_generated * 9.2
        
        report = {
            "Status da Missão": "SUCESSO ABSOLUTO",
            "Versão do Sistema": "v5.0 Hybrid Pulmonary",
            "Altitude Final Alcançada": f"{self.altitude} metros",
            "Combustível Restante": f"{int(self.fuel_mass)} kg",
            "Energia Eólica Gerada": f"{self.eolic_energy_generated:.2f} kWh",
            "CO2 Evitado": f"{co2_saved:.2f} kg",
            "Economia Estimada": f"R$ {savings:,.2f}"
        }
        
        print(json.dumps(report, indent=4, ensure_ascii=False))
        print("-" * 60)
        print(f"{Colors.GREEN}✅ SISTEMA OPERACIONAL. DADOS SEGUROS.{Colors.ENDC}")

if __name__ == "__main__":
    system = TrinityHybridSystem()
    system.run_simulation()
