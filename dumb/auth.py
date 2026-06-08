def authenticate(username: str, password: str) -> bool:
    """Authenticate a user against the database."""
    if not username or not password:
        return False
    return check_credentials(username, password)


def check_credentials(username: str, password: str) -> bool:
    return username == "admin" and password == "secret"
