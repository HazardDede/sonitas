from invoke import Collection

from tasks import (
    config,
    linting,
    testing
)

ns = Collection()

lint = Collection.from_module(linting, name="lint")
test = Collection.from_module(testing, name="test")

# Subtasks
ns.add_collection(lint)
ns.add_collection(test)

# Tasks
ns.add_task(config.config)
