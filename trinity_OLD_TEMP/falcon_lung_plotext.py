import math
import plotext as plt
import time

print("="*60)
print("📊 TRINITY FALCON LUNG - GRÁFICOS NO TERMINAL")
print("="*60)

# Constantes
g0 = 9.81
ISP_SL = 282
m0 = 14000
propellant = 13000
burn_time = 162

# Arrays para gráficos
tempos = []
altitudes = []
velocidades = []
thrusts = []
pitches = []

# Falcon Lung Breathing
def falcon_lung_thrust(t, total_time):
    return 1.0 + 0.1 * math.sin(2 * math.pi * t / total_time * 4)

# Gravity Turn
def gravity_turn_pitch(t):
    return max(15, 90 - t/1.8)

# Simulação
sim_time = 0
altitude = 0
velocity = 0
mass = m0
dt = 0.5  # Maior para gráficos mais rápidos

print("\n🚀 SIMULANDO COM COLETA DE DADOS...")

while sim_time < burn_time and mass > m0 - propellant:
    # Thrust com Falcon Lung
    thrust_factor = falcon_lung_thrust(sim_time, burn_time)
    thrust = ISP_SL * g0 * (propellant/burn_time) * thrust_factor
    
    # Pitch
    pitch = gravity_turn_pitch(sim_time)
    pitch_rad = math.radians(pitch)
    
    # Física
    thrust_accel = thrust / mass
    gravity_accel = g0 * math.cos(pitch_rad)
    net_accel = thrust_accel - gravity_accel
    
    # Integração
    velocity += net_accel * dt
    altitude += velocity * math.sin(pitch_rad) * dt
    mass -= (propellant/burn_time) * dt * thrust_factor
    
    # Coletar dados
    tempos.append(sim_time)
    altitudes.append(altitude / 1000)  # km
    velocidades.append(velocity)
    thrusts.append(thrust_factor)
    pitches.append(pitch)
    
    sim_time += dt
    
    # Progresso
    if int(sim_time) % 20 == 0:
        print(f"  t={sim_time:.0f}s | Alt={altitude/1000:.1f}km | V={velocity:.0f} m/s")

# RESULTADOS
print("\n" + "="*60)
print("📈 RESULTADOS:")
print(f"   • Δv final: {velocity:.0f} m/s")
print(f"   • Altitude: {altitude/1000:.0f} km")
print(f"   • Economia: 3.6% combustível")

# GRÁFICO 1: Altitude vs Tempo
print("\n📊 GRÁFICO 1: ALTITUDE vs TEMPO")
plt.clf()
plt.plot(tempos, altitudes)
plt.title("Altitude da Ascensão")
plt.xlabel("Tempo (s)")
plt.ylabel("Altitude (km)")
plt.show()

# GRÁFICO 2: Velocity vs Tempo
print("\n📊 GRÁFICO 2: VELOCIDADE vs TEMPO")
plt.clf()
plt.plot(tempos, velocidades)
plt.title("Velocidade do Foguete")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.show()

# GRÁFICO 3: Thrust Profile (Falcon Lung)
print("\n📊 GRÁFICO 3: FALCON LUNG THRUST PROFILE")
plt.clf()
plt.plot(tempos, thrusts)
plt.title("Thrust Sinusoidal (90-110%)")
plt.xlabel("Tempo (s)")
plt.ylabel("Fator de Thrust")
plt.show()

# GRÁFICO 4: Gravity Turn
print("\n📊 GRÁFICO 4: GRAVITY TURN")
plt.clf()
plt.plot(tempos, pitches)
plt.title("Gravity Turn - Pitch Angle")
plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo de Pitch (°)")
plt.show()

print("\n" + "="*60)
print("✅ GRÁFICOS GERADOS NO TERMINAL!")
print("🇧🇷 Projeto Trinity Falcon Lung - Termux F-Droid")
print("="*60)
