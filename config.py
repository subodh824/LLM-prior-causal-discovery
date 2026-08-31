import os

class Config:
    RANDOM_SEED = 42

    EPS = 1e-9

    PERFECT_EDGE_PROB = 0.9

    DISTRIBUTION_TO_CUSTOMER = 'distributor_to_customer'
    MANUFACTURER_TO_DISTRIBUTOR = 'manufacturer_to_distributor'
    SUPPLIER_TO_MANUFACTURER = 'supplier_to_manufacturer'
    DATACO = 'dataco'
    SUPPLY_CHAIN_LEGS = [DISTRIBUTION_TO_CUSTOMER, DATACO, MANUFACTURER_TO_DISTRIBUTOR, SUPPLIER_TO_MANUFACTURER]

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    REAL_DATA_DIR = os.path.join(BASE_DIR, 'data')

    EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434/api/chat')

    LLM = {
        'ollama_llama': {
            'host': OLLAMA_HOST,
            'model': 'llama3.1:8b',
            'temperature': 0.7,
            'max_tokens': 1024,
        },
        'ollama_qwen': {
            'host': OLLAMA_HOST,
            'model': 'qwen2.5:1.5b',
            'temperature': 0.7,
            'max_tokens': 1024,
        },
        'gemini': {
            'host': "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            'api_key': os.environ.get('GEMINI_API_KEY'),
            'model': 'gemini-2.5-flash',
            'temperature': 0.7,
            'max_tokens': 16000,
        },
        'cerebras': {
            'host': 'https://api.cerebras.ai/v1/chat/completions',
            'api_key': os.environ.get('CEREBRAS_API_KEY'),
            'model': 'gpt-oss-120b',
            'temperature': 0.7,
            'max_tokens': 16000,
        },
    }

    N_ANOMALOUS = 25
    SHOCK_SDS = 3.0
    DISTRIBUTION_SAMPLES = 800