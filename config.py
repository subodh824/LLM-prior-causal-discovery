import os

class Config:
    RANDOM_SEED = 42

    EPS = 1e-9

    PERFECT_EDGE_PROB = 0.9

    DISTRIBUTION_TO_CUSTOMER = 'distributor_to_customer'
    MANUFACTURER_TO_DISTRIBUTOR = 'manufacturer_to_distributor'
    SUPPLIER_TO_MANUFACTURER = 'supplier_to_manufacturer'
    DATACO = 'dataco'
    SUPPLY_CHAIN_LEGS = [DISTRIBUTION_TO_CUSTOMER,DATACO, MANUFACTURER_TO_DISTRIBUTOR, SUPPLIER_TO_MANUFACTURER ]

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    REAL_DATA_DIR = os.path.join(BASE_DIR, 'data')

    EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

    LLM = {
        'ollama_llama': {
            'host': 'http://localhost:11434/api/chat',
            'model': 'llama3.1:8b',
            'temperature': 0.7,
            'max_tokens': 1024,
        },
        'ollama_qwen': {
            'host': 'http://localhost:11434/api/chat',
            'model': 'qwen2.5:1.5b',
            'temperature': 0.7,
            'max_tokens': 1024,
        },
        'gemini': {
            'host': "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            'api_key': 'REDACTED',
            'model': 'gemini-2.5-flash',
            'temperature': 0.7,
            'max_tokens': 16000,
        },
        'groq': {
            'host': 'https://api.groq.com/openai/v1/c hat/completions',
            'api_key': 'REDACTED',
            'model': 'llama-3.3-70b-versatile',
            'temperature': 0.7,
            'max_tokens': 16000,
        },
        'cerebras': {
            'host': 'https://api.cerebras.ai/v1/chat/completions',
            'api_key': 'REDACTED',
            'model': 'gpt-oss-120b',
            'temperature': 0.7,
            'max_tokens': 16000,
        },
    }


    N_ANOMALOUS = 25
    SHOCK_SDS = 3.0 
    DISTRIBUTION_SAMPLES = 800
