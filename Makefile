test:
	pytest tests -l -vv

format:
	black .

check:
	ruff check . && python -m mypy.dmypy check . && black . --check

check-pylint:
	pylint llmio examples
