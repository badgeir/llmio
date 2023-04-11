test:
	pytest tests

format:
	black .

check:
	ruff . && pylint llmio examples && python -m mypy . && black . --check
