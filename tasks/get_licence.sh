#!/bin/bash
uv run pip-licenses -f csv > all_licenses.csv
uv tree --no-dev | \
    sed -n 's/^[^a-zA-Z0-9]*\([a-zA-Z0-9]\+\).*/\1/p' | head -n -1 > prod_dependencies.txt
grep -f prod_dependencies.txt all_licenses.csv > prod_licenses.csv
