benchmark-complex:
	python3 run.py -s complex -b self-orchestrated -m openai/gpt-4o-mini -j --out complex.json

benchmark-complex-sagepp:
	python3 run.py -s complex -b sage++ -m openai/gpt-4o-mini -j --out complex_sagepp.json

benchmark-simple:
	python3 run.py -s simple -b self-orchestrated -m openai/gpt-4o-mini -j --out simple.json

benchmark-simple-sagepp:
	python3 run.py -s simple -b sage++ -m openai/gpt-4o-mini -j --out simple_sagepp.json

clean:
	docker compose stop opaca-llm-backend
	docker compose rm -f opaca-llm-backend
	docker rmi opaca-llm-backend 2>/dev/null || true