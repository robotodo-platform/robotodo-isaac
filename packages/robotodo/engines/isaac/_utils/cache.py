
# TODO

from typing import Callable, Generic, TypeVar, Hashable


_T = TypeVar("_T")


class CachableRef(Generic[_T]):
    def __init__(
        self, 
        value: _T, 
        key_func: Callable[[_T], Hashable] | None = None,
    ):
        self.value = value
        self.key_func = key_func

    @property
    def key(self):
        return (
            self.key_func(self.value) 
            if self.key_func is not None else 
            id(self.value)
        )
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other: "CachableRef"):
        return (self.key == other.key)