from invoke import Collection

from tasks import (
    config,
    linting
)

ns = Collection()

lint = Collection.from_module(linting, name="lint")

# Subtasks
ns.add_collection(lint)

# Tasks
ns.add_task(config.config)
