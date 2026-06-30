# Fintilligence

A Multi-Agent Finance Intelligence Chatbot powered by LangGraph and Google Gemini.

## Project Structure

```
├── src/
│ ├── agents/       # Agent definitions and logic
│ ├── core/         # Core infrastructure (config, logging)
│ ├── data/         # Data handling and processing
│ ├── rag/          # RAG (Retrieval Augmented Generation) components
│ ├── web_app/      # Web interface components
│ ├── utils/        # Utility functions
│ └── workflow/     # LangGraph workflow definitions
├── tests/          # Test suite
├── config.yaml     # Configuration settings
├── requirements.txt # Project dependencies
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd fintilligence
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    - Copy `.env.example` to `.env` (create one if needed) and set your `GOOGLE_API_KEY`.
    - Review `config.yaml` for model settings.

## Usage

### Using Make (Recommended)

- **Install**: `make install`
- **Run App**: `make run`
- **Test**: `make test`

### Manual Run
```bash
streamlit run src/web_app/app.py
```