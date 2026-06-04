test:
	pytest tests -l -vv

fix:
	ruff check --fix && ruff format

check:
	ruff check . && ruff format --check && python -m mypy.dmypy check .

check-pylint:
	pylint llmio examples

requirements:
	uv export --no-dev > requirements.txt
	uv export --only-group dev > requirements-dev.txt
