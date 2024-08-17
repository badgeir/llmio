test:
	pytest tests -l -vv

format:
	black .

check:
	ruff check . && pylint llmio examples && python -m mypy . && black . --check

requirements:
	poetry export > requirements.txt
	poetry export --only dev > requirements-local.txt
