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


    LLM = {
        # 'ollama': {
        #     'host': 'http://localhost:11434',
        #     'model': 'llama3.1:8b',
        #     'temperature': 0.0,
        #     'max_tokens': 1024,
        # },
        'gemini': {
            'api_key': 'REDACTED',
            'model': 'gemini-2.5-flash',
            'temperature': 0.0,
            'max_tokens': 4096,
        },
        
        'groq': {
            'api_key': 'REDACTED',
            'model': 'llama-3.3-70b-versatile',
            'temperature': 0.0,
            'max_tokens': 4096,
        },
        # 'cerebras': {
        #     'api_key': 'REDACTED',
        #     'model': 'gpt-oss-120b',
        #     'temperature': 0.0,
        #     'max_tokens': 16000,
        # },
    }

    DELIVERY_RISK = {
        DISTRIBUTION_TO_CUSTOMER: {
            "outcome": "late_delivery_risk",
            "causes": ["fulfillment_center_congestion", "weather_severity"],
        },
        MANUFACTURER_TO_DISTRIBUTOR: {
            "outcome": "delivery_to_distributor",
            "causes": ["machine_downtime", "demand_volatility"],
        },
        SUPPLIER_TO_MANUFACTURER: {
            "outcome": "material_delay",
            "causes": [],
        },
    }

    N_ANOMALOUS = 25
    SHOCK_SDS = 3.0 
    DISTRIBUTION_SAMPLES = 800
