from typing import Any, Callable, Dict, Mapping, Tuple
from inspect import Parameter, signature

from pydantic import BaseModel, create_model
from pydantic.typing import get_all_type_hints
from pydantic.utils import to_camel


def model_from_function(function: Callable):

    parameters: Mapping[str, Parameter] = signature(function).parameters

    type_hints = get_all_type_hints(function)
    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, param in parameters.items():
        if name == "state":
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

    return create_model(to_camel(function.__name__), __base__=BaseModel, **fields)
