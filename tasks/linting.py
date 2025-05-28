from invoke import task

from tasks.config import (
    DOCS_PATH,
    SCRIPTS_PATH,
    SOURCE_PATH
)


@task
def flake8(ctx):
    """Runs flake8 linter against codebase."""
    ctx.run("flake8 "
            "--max-line-length 120 "
            "--ignore=E704,E722,E731,W503 "
            f"{SOURCE_PATH}")


@task
def pylint(ctx):
    """Runs pylint linter against codebase."""
    ctx.run(f"pylint {SOURCE_PATH}")


@task
def mypy(ctx):
    """Runs the mypy typing linter against the codebase."""
    ctx.run(f"mypy --ignore-missing-imports {SOURCE_PATH}")


@task(flake8, pylint, mypy, default=True)
def lint(ctx):
    """Run all configured linters against the codebase."""