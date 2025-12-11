# A4S Evaluation module

# Quickstart for Evaluation module

## How to run within local development environment

### Prerequisites
To run the local development environment, you need first to launch the services containers (database, redis, etc.). Please checkout API repo for instructions on how to do this.

### Configuration of development environment
We use `uv` as environment manager, you can configure python dependencies with the following command:

```bash
uv sync --frozen --group dev
```

<!-- We provide a pre-commit hook to automatically check and format your code before each commit. You can install the pre-commit hooks with the following command:

```bash
uv run pre-commit install
``` -->

### Launching locally the A4S Evaluation API
With the services running, you can now launch the A4S Evaluation API locally.

```bash
uv sync --frozen --group dev
bash tasks/start_api.sh
```

### Launching locally the A4S Evaluation Worker
With the services running, you can now launch the A4S Evaluation Worker locally.

```bash
uv sync --frozen --group dev
bash tasks/start_worker.sh
```

### How to manually run linter
We use ruff for linting. This step is automatically run before each commit if the pre-commit hooks are configured.

To manually run the linter, you can use the following command:

```bash
uv sync --frozen --group dev
uv run ruff check .
uv run ruff format .
```

### How to run tests
To run the unit tests, you can use the following command:

```bash
uv sync --frozen --group test
uv run pytest tests/
```

### How to log and customise logs

We use a single package-wide logger named as the main package: `a4s_api`.

To log, simply import the logger and use it:

```python
from a4s_api.utils import get_logger

get_logger().info("This is an info message")
get_logger().error("This is an error message")
```

You can customize the logging configuration by modifying the `./config/logging.yaml` file.

For instance, in the loggers section, you can customize the level of logging for different loggers, such as the `a4s_api` logger (containing our messages) or the `uvicorn` and `sqlalchemy` loggers.

You can also customize the message format by modifying the `formatters.colored.format` field in the loggers section.
See [official Python documentation on LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes) for a full list of available fields.

Please do not push your local changes, except if necessary. For instance, DEBUG log in `logging.yaml` should not be pushed.
