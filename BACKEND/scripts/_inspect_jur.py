import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from app.database import SessionLocal

db = SessionLocal()
print("--- NULL jurisdiction ---")
for r in db.execute(text("select id, title, source_type from source_documents where jurisdiction is null")).all():
    print(r)
print("--- non-India ---")
for r in db.execute(text(
    "select id, title, jurisdiction from source_documents "
    "where jurisdiction is not null and jurisdiction <> 'India' order by jurisdiction"
)).all():
    print(r)
db.close()
