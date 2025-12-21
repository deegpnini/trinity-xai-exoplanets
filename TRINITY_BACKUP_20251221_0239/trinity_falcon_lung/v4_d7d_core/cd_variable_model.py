# Implementação do Cd variável sugerido pelo Grok

def get_cd_for_mach(mach):
    """Cd variable profile as suggested by Grok"""
    if mach < 0.8:
        return 0.35  # Subsonic
    elif mach < 1.2:
        return 0.5   # Transonic (peak drag)
    else:
        return 0.25  # Supersonic
