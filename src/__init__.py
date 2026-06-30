import warnings

# Suppress Pydantic V1 compatibility warning on Python 3.14
warnings.filterwarnings(
    "ignore", 
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater", 
    category=UserWarning
)
