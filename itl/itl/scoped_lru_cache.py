from collections import namedtuple
from functools import update_wrapper
from threading import RLock

import tensorflow as tf
import numpy as np

__all__ = ['scope', 'clear_all_cached_functions']
_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])
_cached_functions = []


class _HashedSeq(list):
    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(args, kwds, typed,
              kwd_mark = (object(),),
              fasttypes = set([int, str, frozenset, type(None)]),
              sorted=sorted, tuple=tuple, type=type, len=len):
    'Make a cache key from optionally typed positional and keyword arguments'
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    # numpy trick
    key = ((item.tostring() if isinstance(item, np.ndarray)
                            else item) for item in key)
    #return _HashedSeq(key), second hack, trouble with _HashedSeq...
    return ''.join(map(str, map(hash, map(str, key))))


def scoped_lru_cache(scope_name, typed=False):
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable or numpy.ndarray.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    def decorating_function(user_function):

        cache = dict()
        stats = [0, 0]                  # make statistics updateable
                                        # non-locally
        HITS, MISSES = 0, 1             # names for the stats fields
        make_key = _make_key
        cache_get = cache.get           # bound method to lookup key or return
                                        # None
        _len = len                      # localize the global len() function
        lock = RLock()                  # because linkedlist updates aren't
                                        # threadsafe
        root = []                       # root of the circular doubly linked
                                        # list
        root[:] = [root, root, None, None]      # initialize by pointing to
                                                # self
        nonlocal_root = [root]                  # make updateable non-locally
        PREV, NEXT, KEY, RESULT = 0, 1, 2, 3    # names for the link fields

        def wrapper(*args, **kwds):
            # simple caching without ordering or size limit
            key = make_key(args, kwds, typed)
            result = cache_get(key, root)   # root used here as a unique
                                            # not-found sentinel
            if result is not root:
                stats[HITS] += 1
                return result
            with tf.variable_scope(scope_name):
                result = user_function(*args, **kwds)
            cache[key] = result
            stats[MISSES] += 1
            return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(stats[HITS], stats[MISSES], maxsize,
                                  len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                root = nonlocal_root[0]
                root[:] = [root, root, None, None]
                stats[:] = [0, 0]

        wrapper.__wrapped__ = user_function
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return update_wrapper(wrapper, user_function)

    return decorating_function


def scope(*args, **kwargs):
    def decorator(func):
        func = scoped_lru_cache(*args, **kwargs)(func)
        _cached_functions.append(func)
        return func

    return decorator


def clear_all_cached_functions():
    for func in _cached_functions:
        func.cache_clear()

