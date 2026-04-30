# Neural Network Ranking Model for Large Language Model Alignment

This repository contains the code for the simulations and LLM llm_experiment described in the paper *"Neural Network Ranking Model for Large Language Model Alignment"*.

## Project Structure

```
.
├── simulation_1/              # Simulation 1: 1D synthetic data
│   ├── simulation_1.py        # Simulation script
│   └── simulation_1.pdf       # Results/figures
├── simulation_2/              # Simulation 2: 50D synthetic data
│   ├── simulation_2.py        # Simulation script
│   └── simulation_2.pdf       # Results/figures
└── llm_experiment/                # LLM llm_experiment on medical MMLU
    ├── collect_prompt.py      # Collect MMLU questions & generate embeddings
    ├── run_gpt3.py            # Generate GPT-3.5-turbo responses
    ├── run_gpt4.py            # Generate GPT-4o responses
    ├── run_alpaca.py          # Generate Alpaca (LoRA-adapted LLaMA-7B) responses
    ├── run_llama1.py          # Generate base LLaMA-7B responses
    ├── run_llama2.py          # Generate LLaMA-2-7B-Chat responses
    ├── simulation_llm.py      # Neural network ranking on real LLM data
    ├── compare_llms.ipynb     # Pairwise model comparison via GPT-4 judge
    ├── reduce_dimension.ipynb # Dimensionality reduction analysis
    ├── data/                  # Questions (CSV) and model responses (JSON)
    ├── eval/                  # Pairwise comparison results
    └── models/                # Local model checkpoints
```

## Components

### Simulation 1 — 1D Prompt Space (`simulation_1/`)

Synthetic simulation with a **1-dimensional** prompt space. Items are modeled on an Erdos-Renyi comparison graph (n=50 items, edge probability p=0.1). Latent preference functions use `sin(i * pi / 8)` with prompts drawn uniformly from (0.3, 0.8). A neural network ranking model is trained via cross-validation over S=3 folds with T=100 Monte Carlo repetitions.

### Simulation 2 — 50D Prompt Space (`simulation_2/`)

Synthetic simulation with a **50-dimensional** prompt space. Items are modeled on an Erdos-Renyi comparison graph (n=80 items, edge probability p=0.07). Latent preference functions use `tanh(X / sqrt(50)) * sin(i * pi / 8)` with prompts drawn from a multivariate uniform distribution. Same cross-validated neural network ranking framework as Simulation 1.

### LLM Experiment (`llm_experiment/`)

Real-world evaluation of 5 LLMs on medical knowledge tasks from the [MMLU](https://huggingface.co/datasets/cais/mmlu) benchmark across 5 domains (Anatomy, Clinical Knowledge, College Biology, College Medicine, Medical Genetics).

**Pipeline:**

1. **Data collection** — `collect_prompt.py` loads 100 questions per domain and generates embeddings via OpenAI `text-embedding-3-small`
2. **Response generation** — `run_*.py` scripts generate responses from each model
3. **Pairwise evaluation** — `compare_llms.ipynb` uses GPT-4 as a judge across all 10 model pairs per domain
4. **Ranking model** — `simulation_llm.py` applies the neural network ranking model to the real LLM comparison data using 256D reduced embeddings
5. **Analysis** — `reduce_dimension.ipynb` performs dimensionality reduction on question embeddings

## Requirements

### Python Dependencies

- `torch`
- `numpy`
- `scipy`
- `pandas`
- `transformers`
- `datasets`
- `openai`
- `tqdm`

### Hardware & Credentials

- **GPU** required for all simulations and local model inference (8-bit quantized models)
- **OpenAI API key** for GPT-3.5/GPT-4o inference and embeddings — set it in `llm_experiment/.env`:
  ```
  OPENAI_API_KEY=your-key-here
  ```

## Data Format

**Model responses** (`llm_experiment/data/`):
```json
[
  {
    "prompt": "Question: A lesion causing... Please choose among: [...]",
    "response": "paralysis of the facial muscles."
  }
]
```

**Evaluation results** (`llm_experiment/eval/`):
```json
{
  "variable_name1": "gpt3",
  "variable_name2": "gpt4o",
  "GPT4_output": [0.0, 0.0, 1.0, ...]
}
```

## Key Parameters

| Component | Parameter | Value |
|---|---|---|
| Simulation 1 | Items / Edge prob / Samples per pair | 50 / 0.1 / 1500 |
| Simulation 2 | Items / Edge prob / Samples per pair | 80 / 0.07 / 1998 |
| Both simulations | Cross-val folds / Monte Carlo reps | 3 / 100 |
| LLM Experiment | Models / Domains / Questions per domain | 5 / 5 / 100 |
| LLM Experiment | Embedding dim / Judge model | 256 / GPT-4 |
