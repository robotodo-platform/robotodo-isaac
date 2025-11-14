import functools
from typing import Any, Sequence, Iterable

# TODO
import numpy
import bracex
import wcmatch.fnmatch


class PathExpression:
    __slots__ = ["_expr"]

    def __init__(self, expr: "PathExpressionLike"):
        # TODO validate
        if isinstance(expr, self.__class__):
            expr = expr._expr
        self._expr = expr
    
    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._expr!r})"

    # TODO
    # @functools.lru_cache(maxsize=1)
    def _cached_is_concrete_single(self, expr: ...):
        match expr:
            case str():
                return not wcmatch.fnmatch.is_magic(
                    expr, 
                    flags=wcmatch.fnmatch.BRACE,
                )
        return False

    # TODO
    # @functools.lru_cache(maxsize=1)
    def _cached_matcher(self, expr: ...):
        return wcmatch.fnmatch.compile(
            expr, 
            flags=wcmatch.fnmatch.BRACE,
            limit=0,
        )

    @property
    def is_concrete_single(self):
        return self._cached_is_concrete_single(self._expr)

    # TODO keep order
    def resolve(self, paths: Iterable[str]):
        return self._cached_matcher(self._expr).filter(paths)

    def match(self, path: str):
        return self._cached_matcher(self._expr).match(path)

    def expand(self):
        return bracex.expand(self._expr)


PathExpressionLike = PathExpression | str | Sequence[str] | numpy.ndarray[str]


def is_path_expression_like(expr: PathExpressionLike | Any):
    match expr:
        case PathExpression():
            return True
        case str():
            return True
        case _ if isinstance(expr, Sequence):
            return True
        case numpy.ndarray():
            return True
        case _:
            pass
    return False