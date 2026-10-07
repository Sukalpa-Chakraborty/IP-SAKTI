import jwt
import os
import uuid
from dotenv import load_dotenv
from datetime import timedelta,datetime,timezone
from main import Header,HTTPException,Depends
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer

load_dotenv()
ACCESS_MINS=30
REFRESH_DAYS=7
secret_key=os.getenv('secret_key')
Algorithm=os.getenv('Algorithm')
security = HTTPBearer()
#login->ceate_token then home('/auth/me')->verify_token then grant access
def create_token(data:dict,token_type:str,lifetime:timedelta)->str:
    
    
    payload={
        **data,#->subpayload=data.copy()->{email}
        'type':token_type,
        'jti':uuid.uuid4().hex,
        'iat':datetime.now(timezone.utc),
        'exp':datetime.now(timezone.utc)+lifetime,
    }#->{sub:email,type:str,jti:uid for revocation,iat:currnttime,exp:now+30 mins}
    
    token=jwt.encode(payload,secret_key,algorithm=Algorithm)
    return token

def create_accessToken(data:dict)->str:
    return create_token(data,'access',timedelta(minutes=ACCESS_MINS))

def create_refreshToken(data:dict)->str:
    return create_token(data,'refresh',timedelta(days=REFRESH_DAYS))

def decode_token(token: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, secret_key, algorithms=[Algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="Wrong token type")
    return payload

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return decode_token(creds.credentials, "access")

      