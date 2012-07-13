__all__ = ['cgcache']

from _pydnd import w_codegen_cache as codegen_cache

# This is the primary codegen cache used by
# the Python exposure of dynd
cgcache = codegen_cache()
