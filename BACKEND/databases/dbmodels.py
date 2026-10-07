from sqlalchemy import Column,Integer,String
from databases.db import Base


class Clients(Base):
    __tablename__='Clients'
    id=Column(Integer,primary_key=True,index=True)
    Username=Column(String,nullable=True,index=True)
    Email=Column(String,nullable=False,index=True)
    Password=Column(String,nullable=False)