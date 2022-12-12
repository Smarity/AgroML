import functools

def doNotRunItTwice(func):
    prev_call = None

    @functools.wraps(func) # It is good practice to use this decorator for decorators
    def wrapper(*args, **kwargs):
        nonlocal prev_call

        if (args, kwargs) == prev_call:
            pass
        prev_call = args, kwargs
        return func(*args, **kwargs)

    return wrapper