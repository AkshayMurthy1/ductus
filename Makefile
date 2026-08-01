.PHONY: help install-cpu install-gpu test lint demo clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install-cpu: ## Laptop install (pipeline + eval)
	pip install -e ".[cpu,dev]"
	python -m spacy download en_core_web_sm || echo "spaCy model optional; regex scrubbing will be used"

install-gpu: ## GPU box install (training)
	pip install -e ".[gpu,dev]"

test: ## Run unit tests
	pytest -q

lint: ## Ruff check
	ruff check src tests

demo: ## End-to-end CPU dry run on the bundled fixture author (no API, no GPU)
	python -m wlm.cli demo --outdir data/demo

matrix: ## Execute the RESEARCH_BRIEF run matrix (resumable) [GPU]
	python scripts/run_matrix.py

results: ## Assemble Tables 1-4 + the frontier figure from run records
	python scripts/assemble_results.py

clean: ## Remove interim artifacts (keeps data/raw and runs)
	rm -rf data/interim/* data/processed/* data/demo
