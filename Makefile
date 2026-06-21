# LCKJOS003 Makefile for ExperimentA
MAIN_SCRIPT   = ExperimentA.py

# Default Experiment Arguments
ALGO          = dt
NOISE         = 0.0
SEED          = 1
DEVICE        = cuda

CHECKPOINT    = checkpoints/dt_seed_1_best.pt

run:
	$(PYTHON) $(MAIN_SCRIPT) --algo $(ALGO) --noise $(NOISE) --seed $(SEED) --device $(DEVICE)

eval:
	@if [ ! -f $(CHECKPOINT) ]; then \
		echo "CRITICAL: Checkpoint target profile missing at '$(CHECKPOINT)'"; \
		exit 1; \
	fi
	$(PYTHON) $(EVAL_SCRIPT) --checkpoint $(CHECKPOINT) --device $(DEVICE)

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ipynb_checkpoints