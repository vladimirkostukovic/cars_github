# Auto Project

A data analytics system for the Czech automotive market built in **Python** with a clean **OOP** architecture.  
It scrapes listings from sauto, processes them through an ETL pipeline,  
and visualizes the results in **Apache Superset**.

## Features
- Object-oriented modular architecture  
- Config-driven ETL pipeline with logging  
- Built-in sanity checks and unit tests  
- Continuous Integration (lint, type check, tests)  
- Visualization dashboards in Apache Superset  
- MIT License

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .

# copy config and run
cp config/config.json
python -m auto_project