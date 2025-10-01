
import abc

class BaseSpec(abc.ABC):
    @abc.abstractmethod
    def random(self, seed: int | None = None):
        ...

    @abc.abstractmethod
    def validate(self, value):
        ...



from typing import Sequence, Hashable


# TODO optimize
def make_size_mapping(
    dims: Sequence[Hashable],
    sizes: Sequence[int], 
    optional_dims: Sequence[Hashable] = tuple(),
):
    """
    TODO doc
    """
    
    if len(dims) < len(sizes):
        return None

    if len(dims) == len(sizes):
        return dict(zip(dims, sizes))

    for i, dim in enumerate(dims):        
        rest = make_size_mapping(
            dims=dims[i + 1:],
            sizes=sizes[i + 1:], 
            optional_dims=optional_dims,
        )
        if rest is None:
            if dim in optional_dims:
                rest = make_size_mapping(
                    dims=dims[i + 1:],
                    sizes=sizes[i:],
                    optional_dims=optional_dims,
                )
                return {
                    dims[i]: None,
                    **(rest if rest is not None else dict()),
                }
        else:
            if i >= len(sizes):
                if dim in optional_dims:
                    return {
                        dims[i]: None,
                        **rest,
                    }
                return None
            return {
                dims[i]: sizes[i],
                **rest,
            }

# TODO
def test_make_size_mapping():
    assert make_size_mapping(
        dims=("b", "a?"),
        sizes=(1, 2),
        optional_dims=("a?", ),
    ) == {
        "b": 1,
        "a?": 2,
    }

    assert make_size_mapping(
        dims=("b", "a"),
        sizes=(1, 2, 3),
    ) == None

    assert make_size_mapping(
        dims=("b", "a?", "c"),
        sizes=(1, 2),
        optional_dims=("a?", ),
    ) == {"b": 1, "a?": None, "c": 2}

    assert make_size_mapping(
        dims=("b", "a?", "c?"),
        sizes=(1, 2),
        optional_dims=("a?", "c?"),
    ) == {"b": 1, "a?": 2, "c?": None}


import re
from typing import Collection, Hashable, Mapping, NamedTuple


# TODO next: mv Shape
class _TODONextShape:
    """
    Shape.
    """

    sizes: Mapping[Hashable, int | None]
    """Mapping from dimension names to their sizes."""

    optional_dims: Collection[Hashable]
    """Collection of optional dimension names."""

    @property
    def dims(self):
        """
        Dimension names.
        """

        return tuple(self.sizes.keys())

    def __init__(
        self,
        sizes: Mapping[Hashable, int | None],
        optional_dims: Collection[Hashable] = frozenset(),
    ):
        """
        TODO doc
        """

        self.sizes = dict(sizes)
        self.optional_dims = frozenset(optional_dims)

    @classmethod
    def from_expression(cls, expr: str):
        """
        Parse a string into a standard :class:`ShapeExpr`.

        :param expr: The string to parse.
        :return: The parsed shape expression.

        Example:

        .. doctest::

            >>> ShapeExpr.from_expression("a[2] b c?[]")
            ShapeExpr(sizes={'a': 2, 'b': None, 'c?': None}, optional_dims={'c?'})

        """

        _SUBEXPR_PATTERN = re.compile(r"(?P<name>.*?)(?:\[(?P<size>.*)\])?$")

        subexprs = str.split(expr)

        sizes = dict()
        optional_dims = set()

        for subexpr in subexprs:
            m = _SUBEXPR_PATTERN.match(subexpr)
            if m is None:
                raise ValueError(f"Invalid shape subexpression: {subexpr!r}")
            
            name = m.group("name")

            size = None
            match m.group("size"):
                case None:
                    size = None
                case "":
                    size = None
                case s:
                    size = int(s)
            
            sizes[name] = size
            if str.endswith(name, "?"):
                optional_dims.add(name)

        return cls(
            sizes=sizes,
            optional_dims=optional_dims,
        )

    def __str__(self):
        """
        String representation of the shape.

        Example:

        .. doctest::

            >>> str(ShapeExpr({'a': 2, 'b': None, 'c?': None}, optional_dims={'c?'}))
            'a[2] b c?'

        """

        return " ".join(
            f"{name}[{size}]" if size is not None else name
            for name, size in self.sizes.items()
        )
    
    def __repr__(self):
        # TODO
        return f"""{_TODONextShape.__qualname__}({self.sizes}, optional_dims={self.optional_dims})"""
    
    # TODO
    # TODO tuple shape_like make_size_mapping
    def fit(self, shape_like: ...):
        """
        TODO doc
        """

        raise NotImplementedError
    



from typing import Sequence, TypeAlias, Hashable, Mapping


# TODO deprecate???
# TODO frozen
# TODO mapping !!!!
class Shape:
    @staticmethod
    def is_compatible(a: "ShapeLike", b: "ShapeLike") -> bool:
        # TODO
        raise NotImplementedError("is_compatible is not implemented yet.")

    @classmethod
    def from_sizes(
        cls,
        sizes: Sequence[int | None],
        dims: Sequence[Hashable] | None = None,
        optional_dims: Sequence[Hashable] = tuple(),
    ):
        size_mapping = make_size_mapping(
            dims=dims if dims is not None else tuple(range(len(sizes))), 
            sizes=sizes,
            optional_dims=optional_dims,
        )
        if size_mapping is None:
            # TODO better message??
            raise ValueError(f"Invalid size mapping inferred from {sizes}: {size_mapping}")
        return cls(size_mapping)

    # TODO
    def __init__(
        self, 
        size_mapping: Mapping[Hashable, int | None] = dict(),
    ):
        self._size_mapping = dict(size_mapping)

    @property
    def sizes(self):
        return tuple(self._size_mapping.values())

    @property
    def dims(self):
        return tuple(self._size_mapping.keys())

    def __repr__(self):
        return f"{Shape.__qualname__}({self._size_mapping})"
    
    def __iter__(self):
        return iter(self.sizes)
    
    def keys(self):
        return self._size_mapping.keys()
    
    def __getitem__(self, key):
        return self._size_mapping[key]

    def update(self, partial_shape: "ShapeLike", **partial_shape_kwds):
        # TODO doc: update shape and return a new Shape instance!!!!

        new_shape = dict(self)
        new_shape.update(partial_shape, **partial_shape_kwds)
        return self.__class__(new_shape)

    def get(self, key, default=None):
        return self._size_mapping.get(key, default)

    def __eq__(self, other):
        if isinstance(other, Shape):
            return self._size_mapping == other._size_mapping
        elif isinstance(other, Mapping):
            return self._size_mapping == other
        elif isinstance(other, Sequence):
            return tuple(self.sizes) == tuple(other)
        return False

ShapeLike: TypeAlias = Sequence[int | None] | Shape



from typing import NamedTuple, Iterable, Collection, Hashable


# TODO deprecate ######
class DimensionExpr(NamedTuple):

    dims: Iterable[Hashable]
    optional_dims: Collection[Hashable]

    @classmethod
    def parse(cls, expr: str):
        dims = str.split(expr)

        return cls(
            dims=dims,
            optional_dims=tuple(
                dim for dim in dims
                if str.endswith(dim, "?")
            ),
        )

class TestDimensionExpr:
    def test_(self):
        assert DimensionExpr.parse("a abc") == DimensionExpr(["a", "abc"], [])
        assert DimensionExpr.parse("b?     c d") == DimensionExpr(["b?", "c", "d"], ["b?"])
# TODO deprecate ######


import collections
from typing import Any, Iterable, Mapping, Literal, Type, Hashable, Annotated, Collection

import numpy

# TODO !!!
import jax
TensorLike = jax.Array


class TensorSpec(BaseSpec):
    """
    TODO doc

    .. doctest::

        >>> spec = TensorSpec("batch? timestep some_dim data")
        >>> spec.dims
        ('batch?', 'timestep', 'some_dim', 'data')
        >>> spec.optional_dims
        frozenset({'batch?'})

    """

    # TODO next: no more optional dim nonsense - use shape directly
    def __init__(
        self,
        spec_or_expr: "TensorSpec | str | None" = None,
        *,
        shape: ShapeLike | None = None,
        dtype: str | Type | None = None,
    ):
        ...

    def __init__(
        self,
        spec_or_expr: "TensorSpec | str | None" = None,
        *,
        dims: Iterable[Hashable] | None = None,
        optional_dims: Collection[Hashable] | None = None,
        shape: ShapeLike | None = None,
        dtype: str | Type | None = None,
    ):
        """
        TODO doc
        """

        self.dims = None
        self.optional_dims = None
        self.shape = None
        self.dtype = None

        match spec_or_expr:
            case TensorSpec():
                self.dims = spec_or_expr.dims
                self.optional_dims = spec_or_expr.optional_dims
                self.shape = spec_or_expr.shape
                self.dtype = spec_or_expr.dtype
            case str():
                self.dims, self.optional_dims = DimensionExpr.parse(spec_or_expr)
            case None:
                # TODO
                pass
            case _:
                raise TypeError(
                    f"Invalid expression: {spec_or_expr}"
                )

        self.dims = tuple(dims or self.dims or tuple())
        for dim, count in collections.Counter(self.dims).items():
            if count > 1:
                raise ValueError(
                    f"Dimension '{dim}' specified multiple times: {self.dims}"
                )
        self.optional_dims = frozenset(optional_dims or self.optional_dims or tuple())

        # TODO
        if isinstance(shape, Sequence):
            raise NotImplementedError("TODO")

        size_mapping = shape or self.shape or dict()
        self.shape = Shape({dim: size_mapping.get(dim, None) for dim in self.dims})
        self.dtype = dtype or self.dtype or None

    def __repr__(self):
        return (
            f"{TensorSpec.__name__}("
            f"dims={self.dims}, "
            f"optional_dims={self.optional_dims}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}"
            f")"
        )

    def set(
        self, 
        tensor: TensorLike, 
        fill_value: TensorLike,
        indices: Mapping[Hashable, int | slice] = dict(),
        copy: bool = False,
    ):
        # TODO multi backend !!!!
        tensor = numpy.array(tensor, copy=copy)

        tensor[tuple(
            indices.get(dim, slice(None))
            for dim in self.dims
        )] = fill_value

        return tensor
    
    def get(
        self,
        tensor: TensorLike,
        indices: Mapping[Hashable, int | slice] = dict(),
        copy: bool = False,
    ):
        # TODO multi backend !!!!
        tensor = numpy.array(tensor, copy=copy)

        return tensor[tuple(
            indices.get(dim, slice(None))
            for dim in self.dims
        )]

    # TODO arbitrary location??
    def expand_dims(
        self, 
        expr: str | None = None,
        tensor: TensorLike | None = None,
        *,
        dims: Iterable[Hashable] | None = None,
        optional_dims: Collection[Hashable] | None = None,
    ):
        """
        TODO doc

        .. doctest::

            >>> spec = TensorSpec("batch? timestep some_dim data")
            >>> spec.expand_dims("timestep")
            Traceback (most recent call last):
                ...
            ValueError: Dimension 'timestep' specified multiple times: ('timestep', 'batch?', 'timestep', 'some_dim', 'data')

        .. seealso ::
        TODO
        """

        if tensor is not None:
            raise NotImplementedError("TODO FIXME WIP")
        
        dims_, optional_dims_ = tuple(), tuple()
        if expr is not None:
            dims_, optional_dims_ = DimensionExpr.parse(expr)
        dims_ = tuple(dims or dims_)
        optional_dims_ = frozenset(optional_dims or optional_dims_)
            
        return TensorSpec(
            dims=(
                *(
                    dims_
                    if self.dims[:len(dims_)] != dims_ else
                    tuple()
                ), 
                *self.dims,
            ),
            optional_dims=(*optional_dims_, *self.optional_dims),
            shape=self.shape,
            dtype=self.dtype,
        )

    def shape_of(self, tensor: TensorLike):
        return Shape.from_sizes(
            numpy.shape(tensor), 
            dims=self.dims, 
            optional_dims=self.optional_dims,
        )

    def reshape(
        self,
        # TODO ShapeLike
        sizes: Mapping[Hashable, int | Literal[-1] | None],
        tensor: TensorLike | None = None,
    ):
        # TODO
        if tensor is None:
            return TensorSpec(
                dims=self.dims,
                optional_dims=self.optional_dims,
                # TODO !!!! check compat!!!!!
                shape=self.shape.update(sizes),
                dtype=self.dtype,
            )
        
        # if (unknown_dims := set(sizes.keys()) - set(self.dims)):
        #     raise ValueError(
        #         f"Unknown dimensions: {unknown_dims}"
        #     )
        
        in_shape = self.shape_of(tensor)
        out_shape = []
        for dim in self.dims:
            # TODO 
            size = sizes.get(dim, in_shape[dim])
            if size is None:
                continue
            out_shape.append(size)
        
        return numpy.reshape(tensor, out_shape)

    # TODO
    def cast(self, tensor: TensorLike):
        ...
        raise NotImplementedError

    # TODO
    def fit(self, tensor: TensorLike):
        dims = self.dims
        optional_dims = self.optional_dims

        shape = self.shape.update(
            self.shape_of(tensor)
        )

        dtype = None
        if self.dtype is not None:
            dtype = numpy.asarray(tensor).dtype
            assert numpy.can_cast(dtype, self.dtype)

        return TensorSpec(
            dims=dims,
            optional_dims=optional_dims,
            shape=shape,
            dtype=dtype,
        )
    
    @property
    def full_shape(self):
        return Shape({
            dim: self.shape[dim] if self.shape[dim] is not None else 1
            for dim in self.shape.dims
        })

    # TODO
    def empty(self): 
        # TODO use everywhere
        # if (unknown_dims := set(default_sizes.keys()) - set(self.dims)):
        #     raise ValueError(
        #         f"Unknown dimensions: {unknown_dims}"
        #     )

        # TODO
        return numpy.empty(
            self.full_shape.sizes, 
            dtype=self.dtype,
        )

    # TODO !!!!!
    def random(
        self, 
        seed: int | None = None,
    ):
        # TODO
        rng = numpy.random.default_rng(seed)
        
        return numpy.asarray(
            rng.random(self.full_shape.sizes),
            dtype=self.dtype,
        )
    
    # TODO !!!!!
    def validate(self):
        raise NotImplementedError


from typing import Annotated, Mapping

class TensorTableSpec(BaseSpec, Mapping):

    # TODO
    def __init__(
        self,
        specs_or_mapping: "TensorTableSpec | Mapping[Hashable, TensorSpec] | None" = None,
        expr: str | None = None,
        *,
        dims: Iterable[Hashable] | None = None,
        optional_dims: Collection[Hashable] | None = None,
    ):

        self.specs = dict()
        self.dims = tuple()
        self.optional_dims = frozenset()

        match specs_or_mapping:
            case TensorTableSpec():
                self.specs = dict(specs_or_mapping)
                self.dims = specs_or_mapping.dims
                self.optional_dims = specs_or_mapping.optional_dims
            case m if isinstance(m, Mapping):
                self.specs = dict(m)
            case None:
                self.specs = dict()
            case _:
                raise TypeError(
                    f"Invalid specs or mapping: {specs_or_mapping}"
                )

        dims_, optional_dims_ = tuple(), tuple()
        if expr is not None:
            dims_, optional_dims_ = DimensionExpr.parse(expr)

        self.dims = tuple(dims or dims_ or self.dims)
        self.optional_dims = frozenset(optional_dims or optional_dims_ or self.optional_dims)

    def __repr__(self):
        return f"{TensorTableSpec.__name__}({self.specs}, dims={self.dims})"
    
    def __len__(self):
        return len(self.specs)

    def __iter__(self):
        return iter(self.specs)

    def __getitem__(self, key):
        return self.specs[key]

    def fit(self, table: Mapping[Hashable, TensorLike]):
        """
        TODO doc

        """
        res = dict()

        for key, spec in self.specs.items():
            if key not in table:
                res[key] = spec
                continue

            tensor = table[key]
            res[key] = spec.fit(tensor)

        return TensorTableSpec(res, dims=self.dims)

    def expand_dims(
        self,
        expr: str | None = None,
        *,
        dims: Iterable[Hashable] | None = None,
        optional_dims: Collection[Hashable] | None = None,
    ):
        specs_ = dict()

        dims_, optional_dims_ = tuple(), tuple()
        if expr is not None:
            dims_, optional_dims_ = DimensionExpr.parse(expr)
        dims_ = tuple(dims or dims_)
        optional_dims_ = frozenset(optional_dims or optional_dims_)

        for key, spec in self.specs.items():
            specs_[key] = spec.expand_dims(
                dims=dims_,
                optional_dims=optional_dims_,
            )

        return TensorTableSpec(
            specs_,
            dims=(
                *(
                    dims_
                    if self.dims[:len(dims_)] != dims_ else
                    tuple()
                ), 
                *self.dims
            ),
            optional_dims=(*optional_dims_, *self.optional_dims),
        )

    def shape_of(self, table: Mapping[Hashable, TensorLike]):
        size_mapping = dict()

        for key, spec in self.specs.items():
            if key not in table:
                continue

            tensor = table[key]

            tensor_shape = spec.shape_of(tensor)
            for dim in self.dims:
                size = tensor_shape[dim]

                if dim in size_mapping:
                    # TODO
                    if size_mapping[dim] != size:
                        raise ValueError(
                            f"Dimension '{dim}' has conflicting sizes: "
                            f"{size_mapping[dim]} vs {size}"
                        )
                size_mapping[dim] = size

        return Shape(size_mapping)

    def reshape(
        self,
        # TODO ShapeLike
        sizes: Mapping[Hashable, int | Literal[-1] | None],
        table: Mapping[Hashable, TensorLike] | None = None,
    ):
        if table is None:
            return TensorTableSpec(
                {
                    key: spec.reshape(sizes)
                    for key, spec in self.specs.items()
                }, 
                dims=self.dims, 
                optional_dims=self.optional_dims,
            )

        res = dict()

        for key, spec in self.specs.items():
            if key not in table:
                continue

            tensor = table[key]
            res[key] = spec.reshape(sizes, tensor=tensor)

        return res

    def empty(self):
        res = dict()

        for key, spec in self.specs.items():
            res[key] = spec.empty()

        return res
    
    def random(
        self, 
        seed: int | None = None,
    ):
        # TODO
        return {
            key: spec.random(seed=seed)
            for key, spec in self.specs.items()
        }
    
    # TODO 
    def validate(self, table: Mapping[Hashable, TensorLike]):
        raise NotImplementedError


from typing import Protocol, Mapping

class TensorTableLike(Mapping):
    """
    This protocol class describes any data structure that 
    can be converted to a tensor table.

    TODO doc
    
    """

    def __class_getitem__(cls, spec: TensorTableSpec):
        return Annotated[Mapping, spec]


import numpy

def as_tensor(tensor_like: TensorLike):
    if not isinstance(tensor_like, TensorLike):
        tensor_like = numpy.asarray(tensor_like)
    return tensor_like

class BoxSpec(TensorSpec):
    """

    TODO doc
    BoxSpec("h w c", bounds=(0., 1.))

    """

    def __init__(
        self,
        spec_or_expr: TensorSpec | str | None = None,
        *,
        # TODO support tensors
        bounds: tuple[TensorLike | None, TensorLike | None] | None = None,
        # shape: ShapeLike | None = None,
        **tensor_spec_kwds,
    ):
        super().__init__(
            spec_or_expr,
            # shape=shape,
            **tensor_spec_kwds,
        )

        self.bounds = (
            tuple(as_tensor(b) for b in bounds) 
            if bounds is not None else 
            tuple((None, None))
        )

        # TODO
        self.shape = Shape.from_sizes(
            numpy.broadcast_shapes(
                tuple(s if s is not None else 1 for s in self.shape.sizes),
                *(b.shape for b in self.bounds if b is not None),
            ),
            dims=self.shape.dims,
            optional_dims=self.optional_dims,
        )

        # TODO make optional?
        # TODO check shape?
        if not any(b is None for b in self.bounds):
            low, high = numpy.broadcast_arrays(*self.bounds)
            if not numpy.all(low <= high):
                raise ValueError(
                    f"Lower bound must be less than or equal to upper bound. "
                    f"Invalid bounds: {self.bounds}"
                )

    def __repr__(self):
        return f"{BoxSpec.__qualname__}(bounds={self.bounds}, shape={self.shape})"

    def random(
        self, 
        seed: int | None = None,
    ):
        rng = numpy.random.default_rng(seed)

        low, high = self.bounds
        low = -numpy.inf if low is None else low
        high = numpy.inf if high is None else high

        return numpy.asarray(
            rng.uniform(low, high, size=self.full_shape.sizes),
            dtype=self.dtype,
        )
    
    def validate(self):
        raise NotImplementedError

    
# TODO
def rescale(
    target_spec: BoxSpec,
    source_spec: BoxSpec,
    source: TensorLike,
):
    """
    TODO doc

    """

    source_min, source_max = source_spec.bounds
    target_min, target_max = target_spec.bounds
    if any(b is None for b in (source_min, source_max, target_min, target_max)):
        raise ValueError(
            f"Bounds must be defined for both source and target specs. "
            f"Invalid bounds: {source_spec.bounds}, {target_spec.bounds}"
        )
    source = as_tensor(source)
    return (
        (source - source_min) / (source_max - source_min)
            * (target_max - target_min) + target_min
    )
