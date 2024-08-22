"""
A walkthrough of this example can be found in the following notebook:
examples/notebooks/simple_task_manager.ipynb
"""

import asyncio
import os
from typing import Literal

from pydantic import BaseModel
from openai import AsyncOpenAI

import llmio


# Define task model
class Task(BaseModel):
    id: int
    name: str
    description: str
    status: Literal["todo", "doing", "done"] = "todo"


TASKS: list[Task] = []


# Define an agent
agent = llmio.Agent(
    instruction="You are a task manager.",
    client=AsyncOpenAI(api_key=os.environ["OPENAI_TOKEN"]),
)


# Define a tool to list tasks
@agent.tool
def list_tasks():
    print("** Listing tasks")
    return TASKS


# Define a tool to add a task
@agent.tool
def add_task(
    name: str, description: str, status: Literal["todo", "doing", "done"] = "todo"
):
    print(f"** Adding task '{name}' with status '{status}'")
    task_id = len(TASKS) + 1
    TASKS.append(Task(id=task_id, name=name, description=description, status=status))
    return "Created task with ID " + str(task_id)


# Define a tool to update a task
@agent.tool
def update_task(
    task_id: int,
    status: Literal["todo", "doing", "done"] | None = None,
    description: str | None = None,
):
    print(f"** Updating task {task_id}")
    for task in TASKS:
        if task.id == task_id:
            task.status = status or task.status
            task.description = description or task.description
            return "Updated task status on task " + str(task_id)
    return "Task not found"


# Define a tool to remove a task
@agent.tool
def remove_task(task_id: int):
    print(f"** Removing task {task_id}")
    for task in TASKS[:]:
        if task.id == task_id:
            TASKS.remove(task)
            return f"OK - removed task {task_id}"
    return "Task not found"


async def main() -> None:
    history: list[llmio.Message] = []

    while True:
        agent_messages, history = await agent.speak(input(">>> "), history=history)

        for message in agent_messages:
            print(f"** Bot: {message}")


if __name__ == "__main__":
    asyncio.run(main())
