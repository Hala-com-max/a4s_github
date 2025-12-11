#!/bin/bash

uv run ruff check .
uv run ruff format .
uv run pytest
