from typing import Any, Callable, Dict, Mapping, Tuple
from inspect import Parameter, signature

from pydantic import create_model, BaseModel
from pydantic.typing import get_all_type_hints
from pydantic.utils import to_camel


def model_from_function(function: Callable) -> type[BaseModel]:
    parameters: Mapping[str, Parameter] = signature(function).parameters

    type_hints = get_all_type_hints(function)
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

    class Config:
        @staticmethod
        def schema_extra(schema: Dict[str, Any]) -> None:
            schema.pop("title")
            for prop in schema.get("properties", {}).values():
                prop.pop("title", None)

    return create_model(to_camel(function.__name__), __config__=Config, **fields)  # type: ignore
