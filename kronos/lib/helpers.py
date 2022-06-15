from datetime import datetime
from functools import wraps
from time import process_time as ptime

def make_date(datetime_str, datetime_format="%Y-%m-%d %H:%M:%S"):
    """ Only return the date part """
    return datetime.strptime(datetime_str, datetime_format).date()

def make_datetime(datetime_str, datetime_format="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(datetime_str, datetime_format)


def delta_seconds(datetime_past, datetime_future):
    return (datetime_future - datetime_past).seconds

def time_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = ptime()
        result = func(*args, **kwargs)
        print("ExecTime:", func.__name__, round(ptime()-s,3), 'sec')
        return result
    return wrapper
