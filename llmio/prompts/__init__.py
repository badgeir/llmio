import textwrap


DEFAULT_COMMAND_HEADER = (
    textwrap.dedent(
        """
    The following commands can be used.
    If you intend to execute a command, only write a valid command and nothing else.
    Do not try to both speak and execute a command at the same time,
    as it will not be accepted as a command.
    Also do not try to execute multiple commands at once.
    You can chain commands, but if so, only execute one command at a time,
    and then execute the next commands afterward.
    Every time a command is executed, the results will be shown as a system message,
    and you then get to either execute a new command or output a normal message
    intended to the user.
    Every time you return a normal text, this will stop the command iteration,
    and the text will be shown to the user. Because of this, do not hint that
    you will execute a command by saying something like "Ok, I will now do X".
    Instead, first execute the command, and then write a normal message to the user.
    Do not talk explicitly about the commands to the user,
    these are hidden and only serve as your interface to the application backend.

    Remember to always EITHER output a command (in json), or a normal message,
    never both at the same time.
"""
    )
    .strip()
    .replace("\n", " ")
)


DEFAULT_COMMAND_PROMPT = textwrap.dedent(
    """
    Command: {{name}}
    {% if description %}
    Description:
    {{description}}
    {% endif %}
    {% if params %}

    Parameters:
    | Name | Type | Description | Required |
    | ---- | ---- | ----------- | -------- |
    {% for param_name, param_type, param_desc, param_required in params %}
    | {{param_name}} | {{param_type}} | {{param_desc}} | {{param_required}} |
    {% endfor %}
    {% endif %}

    Returns:
    | Name | Type | Description |
    | ---- | ---- | ----------- |
    {% for res_name, res_type, res_desc in returns %}
    | {{res_name}} | {{res_type}} | {{res_desc}} |
    {% endfor %}

    Example usage:
    {{mock_data}}

    ---
"""
).strip()


DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    {{description}}

    {{command_header}}

    {% for command in commands %}
    {{command.explain()}}
    {% endfor %} \

    System parameters:
    The current time is {{current_time}}

    You are limited to only answer question regarding the scope described above
    and the available commands defined below.
    For all other questions, politely decline to answer.
"""
).strip()
