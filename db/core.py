from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Use SQLite instead of PostgreSQL - no server needed
import os
# Get the absolute path to the database file
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'billions.db')
engine = create_engine(f'sqlite:///{db_path}', echo=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()
