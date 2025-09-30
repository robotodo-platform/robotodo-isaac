import asyncio
import concurrent.futures


def as_concurrent_future(future: concurrent.futures.Future | asyncio.Future):
    """
    Wrap :class:`concurrent.futures.Future`/:class:`asyncio.Future` 
    into :class:`concurrent.futures.Future`.

    :param future: The asyncio/concurrent future to wrap.
    :return: The concurrent future wrapping the given future.
    """

    match future:
        case concurrent.futures.Future():
            return future

        case asyncio.Future():
            cfut = concurrent.futures.Future()

            def _done_callback(a_future: asyncio.Future):
                try:
                    result = a_future.result()
                except Exception as exc:
                    cfut.set_exception(exc)
                else:
                    cfut.set_result(result)

            future.add_done_callback(_done_callback)
            return cfut

        case _:
            raise TypeError(f"Invalid future: {future}")


# TODO !!!!!!
def as_asyncio_future(future: concurrent.futures.Future | asyncio.Future):
    raise NotImplementedError("TODO")