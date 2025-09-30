"""
TODO

"""


# TODO
import bracex

import wcmatch.fnmatch



class PathExpression:
    # TODO expr: list[str | Path]
    def __init__(self, expr: "str | PathExpression"):
        self._expr = str(expr)

    def __str__(self):
        return self._expr

    # TODO
    def resolve(self, paths: list[str]):
        # TODO cache?
        # TODO brace expansion?
        match = wcmatch.fnmatch.compile(self._expr, limit=0)
        return match.filter(paths)

    # TODO
    def expand(self):
        return bracex.expand(self._expr)



