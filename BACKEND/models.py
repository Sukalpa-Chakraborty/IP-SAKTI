from pydantic import BaseModel,EmailStr

#Login in
class Client_in(BaseModel):
    Email:EmailStr
    Password:str

#register
class client_Create(BaseModel):
    Username:str
    Email:EmailStr
    Password:str
    Confirmpassword:str
