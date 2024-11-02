import textwrap

from llmio import Agent, OpenAIClient
from llmio.agent import _Tool  # noqa: F401


async def test_schema() -> None:
    agent = Agent(
        instruction="""
            You are a calculator.

            {var1} {var2}
        """,
        client=OpenAIClient(api_key="abc"),
    )

    @agent.tool()
    async def add(num1: float, num2: float) -> float:
        """
        Add two numbers together.
        """
        return num1 + num2

    @agent.tool(strict=True)
    async def multiply(num1: float, num2: float) -> float:
        return num1 * num2

    @agent.variable
    def var1() -> str:
        return "value1"

    @agent.variable
    async def var2() -> str:
        return "value2"

    assert (
        agent.summary()
        == textwrap.dedent(
            """
        Tools:
          - add
            Schema:
              {'description': 'Add two numbers together.',
               'name': 'add',
               'parameters': {'properties': {'num1': {'type': 'number'},
                                             'num2': {'type': 'number'}},
                              'required': ['num1', 'num2'],
                              'type': 'object'}}

          - multiply
            Schema:
              {'description': '',
               'name': 'multiply',
               'parameters': {'additionalProperties': False,
                              'properties': {'num1': {'type': 'number'},
                                             'num2': {'type': 'number'}},
                              'required': ['num1', 'num2'],
                              'type': 'object'},
               'strict': True}
        """
        ).lstrip()
    )

    assert (
        await agent._get_instruction(None) == "You are a calculator.\n\nvalue1 value2"
    )

    assert agent._tool_definitions == [
        {
            "function": {
                "description": "Add two numbers together.",
                "name": "add",
                "parameters": {
                    "properties": {
                        "num1": {
                            "type": "number",
                        },
                        "num2": {
                            "type": "number",
                        },
                    },
                    "required": [
                        "num1",
                        "num2",
                    ],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "",
                "name": "multiply",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {
                        "num1": {
                            "type": "number",
                        },
                        "num2": {
                            "type": "number",
                        },
                    },
                    "required": [
                        "num1",
                        "num2",
                    ],
                    "type": "object",
                },
                "strict": True,
            },
            "type": "function",
        },
    ]
