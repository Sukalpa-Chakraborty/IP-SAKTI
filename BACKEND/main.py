from fastapi import FastAPI,Depends,Form,HTTPException,Header,Request,Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from databases.db import get_db,Base,Engine
from databases.dbmodels import Clients
from auth.authentication import create_accessToken,create_refreshToken,decode_token,verify_token
from models import Client_in,client_Create
from hash.Hashing import Hash,verify_passwd




app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

Base.metadata.create_all(bind=Engine)
REFRESH_COOKIE='refresh_token'
@app.post('/auth/register')
def register(username:str=Form(...),email:str=Form(...),password:str=Form(...),cpassword:str=Form(...),db:Session=Depends(get_db)):
    existing=db.query(Clients).filter(Clients.Email==email).first()
    if existing:
        raise HTTPException(
            status_code=401,
            detail='Email already registered'
        )

    if(password==cpassword):
        user=Clients(
            Username=username,
            Email=email,
            Password=Hash(password)

        )
        try:
            db.add(user)
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=400,
                detail='Email already registered'
            )

    else:
        raise HTTPException(
            status_code=400,
            detail='incorrect username or password'
        )        
    return {'message':'User Account created successfully'}

@app.post('/auth/login')
def login(response:Response,email:str=Form(...),password:str=Form(...),db:Session=Depends(get_db)):
    user=db.query(Clients).filter(Clients.Email==email).first()
    if user is None or verify_passwd(password,user) is False:
        raise HTTPException(
            status_code=401,
            detail='Username or password is incorrect'
        )
    response.set_cookie(
        REFRESH_COOKIE,
        create_refreshToken({'sub':email}),
        max_age=7*24*3600,
        httponly= True,
        secure=False,
        samesite='strict',
        path='/auth'
    )
    return {'access_token':create_accessToken({'sub':email}),'token_type':'bearer'}

@app.post('/auth/refresh')
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail='No refresh token')

    payload = decode_token(token, 'refresh')
    user = db.query(Clients).filter(Clients.Email == payload['sub']).first()
    if user is None:
        raise HTTPException(status_code=401, detail='User no longer exists')

   
    response.set_cookie(
        REFRESH_COOKIE, create_refreshToken({'sub': payload['sub']}),
        max_age=7 * 24 * 3600, httponly=True, secure=False, samesite='strict', path='/auth',
    )
    return {'access_token': create_accessToken({'sub': payload['sub']}), 'token_type': 'bearer'}


@app.get('/auth/me')
def home_page(user=Depends(verify_token)):
    return{
        'message':'secure route',
        'user':user['sub']
    }

@app.post('/auth/forgot-password')
def forgot_passwd():
    return

@app.post('/auth/reset-password')
def reset_passwd():
    return

@app.get('/auth/logout')
def logout(response:Response):
    response.delete_cookie(REFRESH_COOKIE, path='/auth')
    return {'message': 'Signed out'}