"""
AERODYNAMIC ENHANCEMENTS v4.1
Cd variável (Grok suggestion) + Aero Heating
"""

import math

class AeroGrokEnhanced:
    """Modelo aerodinâmico aprimorado com sugestões do Grok"""
    
    @staticmethod
    def get_cd_for_mach(mach: float) -> float:
        """Cd variable profile as suggested by Grok"""
        if mach < 0.8:
            return 0.35  # Subsonic
        elif mach < 1.2:
            return 0.5   # Transonic (peak drag)
        else:
            return 0.25  # Supersonic
    
    @staticmethod
    def dynamic_pressure(velocity: float, altitude: float) -> float:
        """Pressão dinâmica (q = 0.5 * ρ * v²) em Pascals"""
        # Densidade simplificada
        rho0 = 1.225  # kg/m³ at sea level
        scale_height = 8500  # meters
        
        rho = rho0 * math.exp(-altitude / scale_height)
        q = 0.5 * rho * velocity**2
        
        return q  # Pa
    
    @staticmethod
    def aero_heating_rate(velocity: float, altitude: float, 
                         nose_radius: float = 0.5) -> float:
        """
        Taxa de aquecimento aerodinâmico simplificado
        Baseado em modelo de reentrada simplificado
        
        Returns: Heat flux in W/m²
        """
        if altitude > 80000:  # Above 80km, negligible
            return 0
        
        # Pressão dinâmica
        q = AeroGrokEnhanced.dynamic_pressure(velocity, altitude)
        
        # Fator de correção Mach
        mach = velocity / 340 if velocity > 0 else 0
        mach_factor = 1.0
        
        if mach > 5:
            mach_factor = 2.0
        elif mach > 3:
            mach_factor = 1.5
        
        # Fator de correção altitude
        alt_factor = 1.0 - min(0.9, altitude / 80000)
        
        # Fator raio do nariz (maior raio = menor heating)
        radius_factor = 1.0 / math.sqrt(nose_radius)
        
        # Heat flux estimado (modelo simplificado)
        # Base: ~100 kW/m² para reentrada severa
        heat_flux = q * 0.1 * mach_factor * alt_factor * radius_factor
        
        return max(0, heat_flux)  # W/m²
    
    @staticmethod
    def total_heat_load(trajectory: list) -> float:
        """
        Calcula carga térmica total durante ascensão
        
        trajectory: lista de tuples (time, velocity, altitude)
        Returns: Total heat load in kJ/m²
        """
        total_heat = 0.0
        
        for i in range(len(trajectory) - 1):
            t1, v1, alt1 = trajectory[i]
            t2, v2, alt2 = trajectory[i + 1]
            
            # Valores médios no intervalo
            v_avg = (v1 + v2) / 2
            alt_avg = (alt1 + alt2) / 2
            dt = t2 - t1
            
            # Heat rate médio
            heat_rate = AeroGrokEnhanced.aero_heating_rate(v_avg, alt_avg)
            
            # Heat no intervalo
            heat_increment = heat_rate * dt  # J/m²
            
            total_heat += heat_increment
        
        return total_heat / 1000  # converte para kJ/m²

class AeroAnalysis:
    """Análise aerodinâmica completa"""
    
    @staticmethod
    def run_analysis(velocity_profile: list, altitude_profile: list, 
                    time_profile: list) -> dict:
        """Executa análise aerodinâmica completa"""
        
        # Combinar trajetória
        trajectory = list(zip(time_profile, velocity_profile, altitude_profile))
        
        # Pressão dinâmica máxima
        q_values = [AeroGrokEnhanced.dynamic_pressure(v, alt) 
                   for v, alt in zip(velocity_profile, altitude_profile)]
        max_q = max(q_values) if q_values else 0
        max_q_time = time_profile[q_values.index(max_q)] if q_values else 0
        
        # Heating máximo
        heat_rates = [AeroGrokEnhanced.aero_heating_rate(v, alt) 
                     for v, alt in zip(velocity_profile, altitude_profile)]
        max_heat = max(heat_rates) if heat_rates else 0
        max_heat_time = time_profile[heat_rates.index(max_heat)] if heat_rates else 0
        
        # Carga térmica total
        total_heat = AeroGrokEnhanced.total_heat_load(trajectory)
        
        # Mach máximo
        mach_values = [v / 340 for v in velocity_profile]
        max_mach = max(mach_values) if mach_values else 0
        
        # Análise de regimes
        regimes = {
            "subsonic": sum(1 for m in mach_values if m < 0.8),
            "transonic": sum(1 for m in mach_values if 0.8 <= m < 1.2),
            "supersonic": sum(1 for m in mach_values if m >= 1.2)
        }
        
        return {
            "max_dynamic_pressure_kpa": max_q / 1000,
            "max_q_time_s": max_q_time,
            "max_heat_flux_kwpm2": max_heat / 1000,
            "max_heat_time_s": max_heat_time,
            "total_heat_load_kjpm2": total_heat,
            "max_mach": max_mach,
            "regime_counts": regimes,
            "q_at_max_velocity": q_values[-1] / 1000 if q_values else 0,
            "heat_at_max_velocity": heat_rates[-1] / 1000 if heat_rates else 0
        }
    
    @staticmethod
    def generate_report(analysis: dict) -> str:
        """Gera relatório de análise aerodinâmica"""
        
        report = []
        report.append("=" * 60)
        report.append("📊 AERODYNAMIC ANALYSIS REPORT (Grok Enhanced)")
        report.append("=" * 60)
        
        report.append(f"\n🎯 CRITICAL VALUES:")
        report.append(f"   • Max Dynamic Pressure: {analysis['max_dynamic_pressure_kpa']:.1f} kPa")
        report.append(f"     (at t = {analysis['max_q_time_s']:.1f} s)")
        report.append(f"   • Max Heat Flux: {analysis['max_heat_flux_kwpm2']:.1f} kW/m²")
        report.append(f"     (at t = {analysis['max_heat_time_s']:.1f} s)")
        report.append(f"   • Total Heat Load: {analysis['total_heat_load_kjpm2']:.0f} kJ/m²")
        
        report.append(f"\n📈 FLIGHT REGIMES:")
        report.append(f"   • Subsonic (< Mach 0.8): {analysis['regime_counts']['subsonic']} points")
        report.append(f"   • Transonic (0.8-1.2): {analysis['regime_counts']['transonic']} points")
        report.append(f"   • Supersonic (> Mach 1.2): {analysis['regime_counts']['supersonic']} points")
        report.append(f"   • Max Mach: {analysis['max_mach']:.2f}")
        
        report.append(f"\n🌡️ THERMAL ANALYSIS:")
        report.append(f"   • Heat at max velocity: {analysis['heat_at_max_velocity']:.1f} kW/m²")
        report.append(f"   • Dynamic pressure at max velocity: {analysis['q_at_max_velocity']:.1f} kPa")
        
        report.append(f"\n💡 RECOMMENDATIONS:")
        if analysis['max_heat_flux_kwpm2'] > 100:
            report.append("   ⚠ Consider TPS (Thermal Protection System)")
        if analysis['max_dynamic_pressure_kpa'] > 35:
            report.append("   ⚠ Max Q constraint may be approached")
        
        report.append("   • Cd profile used: 0.35 (sub) → 0.5 (trans) → 0.25 (super)")
        report.append("   • Model includes: dynamic pressure + aero heating")
        
        report.append("\n" + "=" * 60)
        report.append("✅ ANALYSIS COMPLETE - READY FOR GROK")
        report.append("=" * 60)
        
        return "\n".join(report)
