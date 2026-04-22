PREDICTOR_TYPE ?= habit
BENCHMARK_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: benchmark-complex benchmark-complex-sagepp benchmark-simple benchmark-simple-sagepp analytics slide-metrics clean

benchmark-complex:
	cd $(BENCHMARK_DIR) && python3 run.py -s complex -b self-orchestrated -m openai/gpt-4o-mini -j --out complex.json

benchmark-complex-sagepp:
	cd $(BENCHMARK_DIR) && python3 run.py -s complex -b sage++ -m openai/gpt-4o-mini -j --out complex_sagepp.json --predictor-type $(PREDICTOR_TYPE)

benchmark-simple:
	cd $(BENCHMARK_DIR) && python3 run.py -s simple -b self-orchestrated -m openai/gpt-4o-mini -j --out simple.json

benchmark-simple-sagepp:
	cd $(BENCHMARK_DIR) && python3 run.py -s simple -b sage++ -m openai/gpt-4o-mini -j --out simple_sagepp.json --predictor-type $(PREDICTOR_TYPE)

analytics:
	cd $(BENCHMARK_DIR) && python3 benchmark_results/analytics/reward.py
	cd $(BENCHMARK_DIR) && python3 benchmark_results/analytics/risk.py
	cd $(BENCHMARK_DIR) && python3 benchmark_results/analytics/slide_metrics.py

slide-metrics:
	cd $(BENCHMARK_DIR) && python3 benchmark_results/analytics/slide_metrics.py

clean:
	cd $(BENCHMARK_DIR) && docker compose stop opaca-llm-backend
	cd $(BENCHMARK_DIR) && docker compose rm -f opaca-llm-backend
	cd $(BENCHMARK_DIR) && docker rmi opaca-llm-backend 2>/dev/null || true
