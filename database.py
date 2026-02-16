from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user:password@localhost:5432/dashboard_db"
)

# Création du moteur de base de données
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Session pour les requêtes
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles
Base = declarative_base()

def get_db():
    """Dépendance pour obtenir une session DB dans les routes FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Crée les tables dans la base de données"""
    Base.metadata.create_all(bind=engine)
