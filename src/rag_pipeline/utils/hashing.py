import hashlib
def get_file_hash(file_bytes):
    """Şəklin məzmununa görə unikal MD5 hash yaradır."""
    return hashlib.md5(file_bytes).hexdigest()
