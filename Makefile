## Thesis pipeline — common tasks.
##
## Usage:  make evaluate   (re-score all system outputs)
##         make stats       (regenerate stats report)
##         make plots       (regenerate all plots)
##         make all         (evaluate → stats → plots)
##         make test        (run test suite)

PYTHON := .venv/bin/python
SCHEMA := chr

.PHONY: all evaluate stats plots test clean

all: evaluate stats plots

evaluate:
	@echo "=== Evaluating schema/full ==="
	$(PYTHON) pipeline/evaluate.py --schema $(SCHEMA) --track schema --prompt full
	@echo "=== Evaluating ontology/ontology ==="
	$(PYTHON) pipeline/evaluate.py --schema $(SCHEMA) --track ontology --prompt ontology
	@echo "=== Evaluating free-model schema/full ==="
	OUTPUTS_ROOT_OVERRIDE=$(PWD)/evaluation/outputs_freemodel $(PYTHON) pipeline/evaluate.py --schema $(SCHEMA) --track schema --prompt full
	@echo "=== Evaluating free-model ontology/ontology ==="
	OUTPUTS_ROOT_OVERRIDE=$(PWD)/evaluation/outputs_freemodel $(PYTHON) pipeline/evaluate.py --schema $(SCHEMA) --track ontology --prompt ontology

stats:
	@echo "=== Regenerating stats report ==="
	$(PYTHON) pipeline/stats.py

plots:
	@echo "=== Generating all plots ==="
	$(PYTHON) pipeline/plots.py --all-tracks --vignette vignette_001

test:
	@echo "=== Running test suite ==="
	$(PYTHON) -m pytest tests/ -v

clean:
	find evaluation/outputs/plots -name "*.png" -delete
	find evaluation/outputs/plots -name "*.txt" -delete
	@echo "Cleaned plots directory."
