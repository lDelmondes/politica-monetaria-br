# Efeitos da Política Monetária no Brasil (1999-2025)

Análise dos impactos da taxa Selic sobre o nível de atividade econômica (IBC-Br/PIB) 
e a inflação (IPCA) no Brasil, com modelos VAR e BVAR em frequência mensal.

## Estrutura

- `data/raw/`: dados brutos das fontes (BCB, IBGE, World Bank)
- `data/processed/`: dados tratados, prontos para modelagem
- `notebooks/`: análises em Jupyter
- `src/`: funções reutilizáveis
- `outputs/`: gráficos e tabelas finais

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```