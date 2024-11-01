from typing import Any, Callable, Dict, Mapping, Tuple, get_type_hints
from inspect import Parameter, signature

from pydantic import create_model, BaseModel


def to_camel(string: str) -> str:
    return "".join(word.capitalize() for word in string.split("_"))


def model_from_function(function: Callable) -> type[BaseModel]:
    parameters: Mapping[str, Parameter] = signature(function).parameters
    type_hints = get_type_hints(function, include_extras=True)

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, param in parameters.items():
        if name == "_context":
            continue

        if param.annotation is param.empty:
            annotation = Any
        else:
            annotation = type_hints[name]

        default = ... if param.default is param.empty else param.default
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            fields[name] = annotation, default
        else:
            raise ValueError(
                "Unable to parse function signature. Only named arguments supported."
            )

    model = create_model(to_camel(function.__name__), **fields)  # type: ignore
    return model
