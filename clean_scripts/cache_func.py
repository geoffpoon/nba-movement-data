# =============================================================================
# Check out this site to see what the caching function is based on
# https://datascience.blog.wzb.eu/2016/08/12/a-tip-for-the-impatient-simple-caching-with-python-pickle-and-decorators/
# =============================================================================


import os
import pickle

def cached(cache_file):
    """
    A function that creates a decorator
    which will use "cachefile" for chaching the results
    of the decorated function "fn"
    """
    def decorator(fn):  # def a decorator for func "fn"
        def wrapped(*args, **kwargs):   # def wrapper that will finally call "fn" with all args
            # if cache exists -> load it and return its content
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as cachehandle:
                    print("Using cached result from '%s' " %cache_file)
                    return pickle.load(cachehandle)
            
            # execute "fn" with all args passed
            result = fn(*args, **kwargs)
            
            # write to cache file
            with open(cache_file, 'wb') as cachehandle:
                print("Saving result to cache file '%s' " %cache_file)
                pickle.dump(result, cachehandle)
                
            return result
        
        return wrapped
    
    return decorator    # return this "customized" decorator that uses "cachefile"


    