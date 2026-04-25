.PHONY: help install test serve baseline train evaluate clean fmt

help:
	@echo "DeceptEnv — common tasks"
	@echo "  make install    install python deps (server + training)"
	@echo "  make test       run pytest smoke suite"
	@echo "  make serve      run the FastAPI env server on :7860"
	@echo "  make baseline   run cover-story baseline (50 eps) — needs server"
	@echo "  make train      run REINFORCE trainer (small Qwen, 50 iters)"
	@echo "  make evaluate   run post-train evaluation + comparison plot"
	@echo "  make clean      remove runs/ + caches"

install:
	pip install -r requirements.txt

test:
	pytest -q tests/

serve:
	python -m server.app

baseline:
	python -m evaluation.baseline --episodes 50 --policy cover

train:
	python -m training.rl_trainer --iterations 50 --episodes-per-iter 4

evaluate:
	@echo "Pass --adapter <path/to/lora> to compare a trained checkpoint." && \
	python -m evaluation.evaluate --model Qwen/Qwen2.5-0.5B-Instruct \
	    --baseline runs/baseline/episodes.json

clean:
	rm -rf runs __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
