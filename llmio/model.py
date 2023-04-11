from typing import Any, Callable, Dict, Mapping, Tuple

from pydantic import BaseModel, create_model
from pydantic.typing import get_all_type_hints
from pydantic.utils import to_camel


def model_from_function(function: Callable):
    from inspect import Parameter, signature

    parameters: Mapping[str, Parameter] = signature(function).parameters

    type_hints = get_all_type_hints(function)
    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, p in parameters.items():
        if name == "system_params":
            continue

        if p.annotation is p.empty:
            annotation = Any
        else:
            annotation = type_hints[name]

        default = ... if p.default is p.empty else p.default
        if p.kind == Parameter.POSITIONAL_OR_KEYWORD:
            fields[name] = annotation, default
        else:
            raise ValueError(
                "Unable to parse function signature. Only named arguments supported."
            )

    return create_model(to_camel(function.__name__), __base__=BaseModel, **fields)
