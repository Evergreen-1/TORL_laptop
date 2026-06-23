# LCKJOS003 Makefile for ExperimentA
PYTHON        = python
MAIN_SCRIPT   = ExperimentA.py

# Default settings (Overridden via CLI)
ALGO          = dt
NOISE         = 0.0
SEED          = 1
DEVICE        = directml
CHECKPOINT    = checkpoints/cql_noise_0.00_seed_0.pt
STEPS		  = 35000
RESUME        = True

.PHONY: help run run-full eval

help:
	@echo "Available commands:"
	@echo "  make run          - Train a single model layout"
	@echo "  make run-full     - Execute complete matrix parameter sweep"
	@echo "  make eval         - Evaluate a saved checkpoint using internal functions"

run:
	$(PYTHON) $(MAIN_SCRIPT) --algo $(ALGO) --noise $(NOISE) --seed $(SEED) --device $(DEVICE) --steps $(STEPS)

run-full:
	$(PYTHON) $(MAIN_SCRIPT) --full

eval:
	$(PYTHON) $(MAIN_SCRIPT) --checkpoint $(CHECKPOINT) --device $(DEVICE) 

resume:
	$(PYTHON) $(MAIN_SCRIPT) --checkpoint $(CHECKPOINT) --resume --device $(DEVICE) --steps $(STEPS)

clean:
	rm -rf __pycache__ */__pycache__ *.pyc *.pyo .pytest_cache