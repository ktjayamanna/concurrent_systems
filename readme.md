# SAGE++: Speculative tool execution for AI agents

## Setup

**Clone the repo**
```bash
git clone git@github.com:ktjayamanna/concurrent_systems.git
cd concurrent_systems
```

**Build the dev container**

Make sure you have created .env file at .devcontainer/.env. Then run,

```bash
docker build -t concurrent_systems_dev .
docker run -it --rm -v $(pwd):/code concurrent_systems_dev bash
```

## Reproducing Benchmarks

```bash
cd src/benchmark
```

| Command | Description |
|---|---|
| `make benchmark-complex` | Runs the complex task suite against the self-orchestrated baseline (GPT-4o-mini) |
| `make benchmark-complex-sagepp` | Runs the complex task suite against the Sage++ system; set `PREDICTOR_TYPE=<type>` to swap predictors (default: `habit`) |
| `make benchmark-simple` | Runs the simple task suite against the self-orchestrated baseline |
| `make benchmark-simple-sagepp` | Runs the simple task suite against Sage++; same `PREDICTOR_TYPE` override applies |
| `make analytics` | Runs all three analytics scripts: **reward.py** (plots prediction time vs. execution time and tool-sequence distributions), **risk.py** (plots risk/cost tradeoff metrics), and **slide_metrics.py** (re-calculates all predictors against the question sets). Use this to regenerate every figure in the paper at once. |
| `make slide-metrics` | Runs only **slide_metrics.py** which re-calculates HabitPredictor, NaiveBayesPredictor, and SmallLLMPredictor against the question sets and writes the predictor-accuracy/hit-rate plots used in the paper slides. Use this when you only changed predictor logic and don't need to replot reward or risk. |
| `make habit-sweep` | Sweeps habit working-set size hyperparameter for the habit predictor |
| `make clean` | Stops and removes the `opaca-llm-backend` Docker container and image |
