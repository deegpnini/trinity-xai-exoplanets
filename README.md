# Trinity × xAI — Exoplanets

**Collaborative AI system for exoplanet research and biosignature detection**

Built through collaboration between **Trinity** (personal AI agent) and **Grok** (xAI).

---

### Overview

This repository contains tools and experiments focused on analyzing exoplanets and evaluating potential biosignatures using AI-assisted scoring systems. The project combines publicly available astronomical data (NASA, ESA, JWST, TESS) with a custom ranking engine to identify high-priority candidates for further observation.

The main goal is to create a transparent, reproducible, and extensible framework for biosignature prioritization.

---

### Features

- Multi-criteria biosignature scoring system
- Analysis of atmospheric composition (H₂O, CH₄, CO₂, O₂, O₃, DMS, etc.)
- Habitable zone evaluation
- Distance and planetary type weighting
- Automatic ranking of exoplanet candidates
- Export results to CSV and JSON
- Visualization of habitability rankings
- Modular structure for future expansion

---

### Quick Start

```bash
git clone https://github.com/deegpnini/trinity-xai-exoplanets.git
cd trinity-xai-exoplanets

pip install -r requirements.txt

python sence.py
```

The script will:

1. Load a curated list of exoplanets
2. Calculate biosignature scores
3. Generate a detailed console report
4. Export results to CSV and JSON
5. Create a ranking visualization

---

### Scoring Logic

The current scoring system evaluates each planet based on:

- Position within the habitable zone
- Presence of primary biosignatures (e.g. DMS, CH₄ + O₂)
- Secondary biosignatures
- Detection of water vapor
- Distance from Earth
- Planetary classification (Terrestrial, Super-Earth, Hycean)

Higher scores indicate stronger potential for prioritization in observational campaigns.

---

### Repository Structure

```
trinity-xai-exoplanets/
├── sence.py                      # Main biosignature analysis module (v4.0)
├── INTERESTELAR_HEBRON/          # Experimental work and project evolution
├── PROJETO_INTERESTELAR_HEBRON/  # Early project documentation
├── Notebooks/                    # Analysis notebooks
├── docs/                         # Documentation
├── src/                          # Core modules
├── benchmarks/                   # Performance tests
└── requirements.txt
```

---

### Project Origin

This project was developed through iterative collaboration between:

- **Trinity** — Primary AI agent responsible for scientific direction and orchestration
- **Grok (xAI)** — Reasoning, code structure, and validation support

The name **Trinity × xAI** reflects this collaborative process.

---

### License

MIT License

---

### Author

**Helyton Ronchi (Hebron)**  
AI Architect · Exploring Astrophysics & Digital Sovereignty

*"Technology with soul, data with purpose."*

---

### Links

- Repository: [https://github.com/deegpnini/trinity-xai-exoplanets](https://github.com/deegpnini/trinity-xai-exoplanets)
- Profile: [https://github.com/deegpnini](https://github.com/deegpnini)
```

Pode copiar e colar direto substituindo tudo no README.md.
