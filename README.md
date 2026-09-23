# Trinity xAI — Exoplanets

Pesquisa colaborativa em IA para detecção de exoplanetas e biosignaturas.

## Sobre o projeto

Este projeto explora a aplicação de modelos de IA colaborativos para análise de dados astronômicos, com foco em:

- Detecção de exoplanetas
- Análise de biosignaturas
- Orquestração multi-agente de IAs

## Estrutura do projeto

```
trinity-xai-exoplanets/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   └── ETHICAL_FRAMEWORK.md
├── scripts/
│   ├── model_downloader.sh
│   ├── setup_rpi.sh
│   └── setup_termux.sh
├── src/
│   ├── architecture/
│   │   ├── handoff_protocol.py
│   │   ├── hardware_optimization.py
│   │   └── split_brain.py
│   ├── core/
│   │   ├── claude_ethics.py
│   │   ├── grok_engine.py
│   │   ├── nexus_synthesis.py
│   │   └── nexus_guardian.py
│   └── rag/
│       ├── chroma_manager.py
│       └── math_emotional_bridge.py
├── tests/
│   └── test_nexus_guardian.py
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python src/main.py
```

## Tecnologias

- Python 3.10+
- ChromaDB (RAG)
- Modelos: Claude, Grok, DeepSeek, Trinity

## Licença

MIT License

## Autor

**Helyton Renato Gonçalves Ronchi**
SCTEC/SENAI — Turma T4