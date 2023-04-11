# llmio
Large Language Model I/O

![pylint](https://github.com/github/docs/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/github/docs/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/github/docs/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)

# Setup

``` python
conda create -n llmio python=3.10
conda activate llmio

pip install -r requirements-local.txt
pip install -r requirements.txt

make check
make format
makt test
```
