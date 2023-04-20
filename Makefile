test:
	pytest tests -l -vv

format:
	black .

check:
	ruff . && pylint llmio examples && python -m mypy . && black . --check
