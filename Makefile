UV ?= uv
PYTHON = $(UV) run --locked python

.PHONY: lint smoke study report stability population ablation learned selection ci-docker
lint:
	$(UV) run --locked black --check cart_study
	$(UV) run --locked isort --check-only cart_study
	$(UV) run --locked ruff check cart_study

smoke:
	@cart_smoke_dir=$$(mktemp -d) || exit $$?; \
	trap 'rm -rf "$$cart_smoke_dir"' EXIT; \
	$(PYTHON) -m cart_study.study --smoke --output "$$cart_smoke_dir/study" && \
	$(PYTHON) -m cart_study.stability --smoke --output "$$cart_smoke_dir/stability" && \
	$(PYTHON) -m cart_study.ablation --smoke --jobs 2 --output "$$cart_smoke_dir/ablation" && \
	$(PYTHON) -m cart_study.learned --smoke --jobs 2 --output "$$cart_smoke_dir/learned" && \
	$(PYTHON) -m cart_study.selection --smoke --jobs 2 --output "$$cart_smoke_dir/selection"

study:
	$(PYTHON) -m cart_study.study

report:
	$(PYTHON) -m cart_study.report

stability:
	$(PYTHON) -m cart_study.stability

population:
	$(PYTHON) -m cart_study.population

ablation:
	$(PYTHON) -m cart_study.ablation

learned:
	$(PYTHON) -m cart_study.learned

selection:
	$(PYTHON) -m cart_study.selection

ci-docker:
	docker run --rm -v "$(CURDIR):/work:ro" -w /work -e UV_PROJECT_ENVIRONMENT=/tmp/cart-venv -e PYTHONDONTWRITEBYTECODE=1 -e RUFF_CACHE_DIR=/tmp/ruff-cache python:3.14-slim sh -c 'pip install uv && uv sync --locked && uv run --locked black --check cart_study && uv run --locked isort --check-only cart_study && uv run --locked ruff check cart_study && uv run --locked python -m cart_study.study --smoke --output /tmp/cart-study && uv run --locked python -m cart_study.stability --smoke --output /tmp/cart-stability && uv run --locked python -m cart_study.ablation --smoke --jobs 2 --output /tmp/cart-ablation && uv run --locked python -m cart_study.learned --smoke --jobs 2 --output /tmp/cart-learned && uv run --locked python -m cart_study.selection --smoke --jobs 2 --output /tmp/cart-selection'
