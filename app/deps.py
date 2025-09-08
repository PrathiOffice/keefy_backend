from fastapi import Depends
from sqlalchemy.orm import Session
from .database import get_db
from .auth import get_current_user

DB = Session
get_db_dep = get_db
get_current_user_dep = get_current_user