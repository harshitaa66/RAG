from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__truncate_error=False)

hashed = pwd_context.hash("mypassword")
print(hashed)
