# GC Chat Log Evaluator

An LLM-as-judge tool for evaluating exported Genesys Cloud chat logs against user-defined goals. Upload your conversation history, define what "success" looks like, and let a locally-hosted LLM classify each conversation's intent and evaluate whether the agent achieved it.

## How It Works

1. **Define goals** in a YAML file — each goal has a name, description, and success/failure criteria
2. **Export chat logs** from Genesys Cloud (CSV from the conversation history view, or JSON)
3. **Run the evaluator** — for each conversation, the LLM:
   - **Classifies** which goal the customer was pursuing (or marks it "unclassified")
   - **Evaluates** whether the agent achieved that goal
4. **Review results** — per-goal success rates, explanations, and full conversation breakdowns

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) running locally with a model pulled (e.g., `ollama pull llama3.2`)

## Setup

```bash
cd gc-chat-log-evaluator
pip install -r requirements.txt
```

## Running the Web UI

```bash
python3 -m src.web_app
```

Open http://localhost:5000 in your browser. Upload:
- **Chat Log File** — a `.csv` export from Genesys Cloud or a `.json` chat log
- **Evaluation Suite File** — a `.yaml` or `.json` file defining your goals
- **Ollama Model** — e.g., `llama3.2`

Results stream live as each conversation is evaluated, showing the conversation, classified goal, and explanation.

## Running via CLI

```bash
python3 -m src.cli chatlog.csv goals.yaml --ollama-model llama3.2
```

### CLI Options

| Flag | Description |
|------|-------------|
| `chat_log` | Path to chat log file (JSON or CSV) |
| `evaluation_suite` | Path to evaluation suite file (YAML or JSON) |
| `--ollama-url` | Ollama base URL (default: http://localhost:11434) |
| `--ollama-model` | Ollama model name (required) |
| `--timeout` | LLM request timeout in seconds (default: 120) |
| `--output-format` | Export format: `csv` or `json` |
| `--output-file` | Write report to file instead of stdout |

## Evaluation Suite Format

```yaml
name: My Evaluation Suite

goals:
  - name: Product Complaint
    description: Customer is reporting a problem with a product
    criteria: >
      The agent's response must contain the exact intent label "Product Complaint".
      SUCCESS: The agent responds with "Product Complaint".
      FAILURE: The agent responds with a different label or a conversational reply.

  - name: Voucher/Refund/Replacement
    description: Customer is requesting a voucher, refund, or replacement
    criteria: >
      The agent's response must contain "Voucher/Refund/Replacement".
      SUCCESS: The agent responds with the correct label.
      FAILURE: The agent responds with a different label.
```

### Goal Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Goal name — used for classification and reporting |
| `description` | Yes | What this goal represents |
| `criteria` | Yes | How to determine success/failure for this goal |

## Chat Log Formats

### Genesys Cloud CSV Export

Export from the Genesys Cloud console → Conversation History. The CSV has columns:

```
Conversation ID, Session ID, Date, Utterance, Prompt, Ask Action Number, ...
```

The parser groups rows by `Conversation ID`, maps `Utterance` → customer messages and `Prompt` → agent messages, and automatically deduplicates repeated "no input" timeout messages.

### Generic JSON

```json
{
  "conversations": [
    {
      "id": "conv-123",
      "messages": [
        {"role": "agent", "content": "Hello! How can I help?"},
        {"role": "customer", "content": "I need help with my order"}
      ]
    }
  ]
}
```

## Configuration

Set defaults via environment variables or a `config.yaml` file:

| Env Variable | Config Key | Description |
|-------------|------------|-------------|
| `OLLAMA_BASE_URL` | `ollama_base_url` | Ollama URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | `ollama_model` | Ollama model name |
| `GC_EVALUATOR_LLM_TIMEOUT` | `llm_timeout` | Request timeout in seconds (default: 120) |

Precedence: Web UI > Environment variables > config.yaml > defaults

## Results

The results page shows:
- **Live progress** as each conversation is evaluated
- **Per-goal success rates** with conversations grouped by classified goal
- **Expandable conversations** showing the full message exchange
- **LLM explanations** for each classification and achievement decision
- **Export** to CSV or JSON (JSON includes full conversation histories)

## Running Tests

```bash
python3 -m pytest tests/ -v
```
