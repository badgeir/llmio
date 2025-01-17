test:
	pytest tests -l -vv

format:
	black .

fix:
	ruff check . --fix

check:
	ruff check . && python -m mypy.dmypy check . && black . --check

check-pylint:
	pylint llmio examples

requirements:
	poetry export > requirements.txt
	poetry export --only dev > requirements-dev.txt
