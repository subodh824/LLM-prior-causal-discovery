import os

class Config:
    RANDOM_SEED = 42

    EPS = 1e-9

    PERFECT_EDGE_PROB = 0.9

    DISTRIBUTION_TO_CUSTOMER = 'distributor_to_customer'
    MANUFACTURER_TO_DISTRIBUTOR = 'manufacturer_to_distributor'
    SUPPLIER_TO_MANUFACTURER = 'supplier_to_manufacturer'
    SUPPLY_CHAIN_LEGS = [DISTRIBUTION_TO_CUSTOMER, MANUFACTURER_TO_DISTRIBUTOR, SUPPLIER_TO_MANUFACTURER]


    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DISTRIBUTION_TO_CUSTOMER_DIR = os.path.join(DATA_DIR, 'distributor_to_customer')
    MANUFACTURER_TO_DISTRIBUTOR_DIR = os.path.join(DATA_DIR, 'manufacturer_to_distributor')
    SUPPLIER_TO_MANUFACTURER_DIR = os.path.join(DATA_DIR, 'supplier_to_manufacturer')

    LLM = {
        # 'ollama': {
        #     'host': 'http://localhost:11434',
        #     'model': 'llama3.1:8b',
        #     'temperature': 0.0,
        #     'max_tokens': 1024,
        # },
        # 'gemini': {
        #     'api_key': 'REDACTED',
        #     'model': 'gemini-2.5-flash',
        #     'temperature': 0.0,
        #     'max_tokens': 4096,
        # },
        
        'groq': {
            'api_key': 'REDACTED',
            'model': 'llama-3.3-70b-versatile',
            'temperature': 0.0,
            'max_tokens': 4096,
        },
        'cerebras': {
            'api_key': 'REDACTED',
            'model': 'gpt-oss-120b',
            'temperature': 0.0,
            'max_tokens': 16000,
        },
    }