from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy import create_engine 
from dotenv import load_dotenv
import os

load_dotenv()


Base=declarative_base()
db_Url=os.getenv('db_url')
Engine=create_engine(db_Url)
sessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=Engine)
def get_db():
    db=sessionLocal()
    try:
        yield db
    finally:
        db.close()
    
