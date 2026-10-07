import bcrypt
from databases.dbmodels import Clients

def Hash(password:str)->str:
    #bcrypt.hashpw->return bytes
    hashed_passwd=bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt()
    )
    return hashed_passwd.decode('utf-8')

def verify_passwd(password:str,user:Clients)->bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'),user.Password.encode('utf-8'))

    except(ValueError,AttributeError):
        return False