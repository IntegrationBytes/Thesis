# KnoBuilder - Knowledge Graph Extraction

A tool for extracting knowledge graphs from unstructured text, featuring a Streamlit UI and a benchmark on the SciERC dataset.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install streamlit langgraph langchain-ollama langchain-openai langchain-core python-dotenv beautifulsoup4 graphviz
    ```
    *(Note: You may need to install Graphviz on your system, e.g., `brew install graphviz` on macOS)*

3.  **Environment Variables:**
    Create a `.env` file for API keys if using cloud models (like Gemini or OpenAI via OpenRouter):
    ```
    OPENROUTER_API_KEY=your_key_here
    ```

## Usage

### 1. Run the Demo Script
Test the extraction logic on a simple sentence:
```bash
python3 demo.py
```

### 2. Run the Interactive UI
Launch the Streamlit app to try different models and inputs:
```bash
streamlit run app.py
```

### 3. Run the SciERC Benchmark
Evaluate the performance against the industry-standard SciERC dataset.

First, download and prepare the data:
```bash
python3 load_scierc.py
```

Then run the benchmark:
```bash
python3 benchmark.py
```

## Benchmark Results

**Model:** `google/gemini-2.0-flash-001`
**Dataset:** SciERC (Scientific Information Extraction)

| Metric | Score |
| :--- | :--- |
| **Precision** | 0.35 |
| **Recall** | 0.34 |
| **F1-Score** | 0.33 |

*Note: Results based on a zero-shot prompt with specific relation definitions.*
