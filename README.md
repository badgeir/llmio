# llmio
Large Language Model I/O

![pylint](https://github.com/badgeir/llmio/actions/workflows/pylint.yml/badge.svg)
![mypy](https://github.com/badgeir/llmio/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/badgeir/llmio/actions/workflows/ruff.yml/badge.svg)
![tests](https://github.com/badgeir/llmio/actions/workflows/test.yml/badge.svg)

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
