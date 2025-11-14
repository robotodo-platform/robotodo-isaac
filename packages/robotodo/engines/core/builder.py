
import abc


class ProtoBuilder(abc.ABC):
    @abc.abstractmethod
    def build(self):
        ...

