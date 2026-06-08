def connect_database(url: str, pool_size: int = 10):
    """Create a database connection pool."""
    return DatabasePool(url, pool_size)


class DatabasePool:
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size

    def query(self, sql: str):
        return []
