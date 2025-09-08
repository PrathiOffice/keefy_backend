# database.py - Centralized database configuration
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from bson import ObjectId

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

# MongoDB connection configuration
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "musicapp")

# Global database instance
mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    mongodb.client = AsyncIOMotorClient(MONGO_URL)
    mongodb.database = mongodb.client[DB_NAME]
    
    # Create indexes for better performance
    await mongodb.database["users"].create_index([("username", ASCENDING)], unique=True)
    await mongodb.database["users"].create_index([("email", ASCENDING)], unique=True)
    await mongodb.database["music"].create_index([("name", ASCENDING)])
    
    print("Connected to MongoDB!")

async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        print("Disconnected from MongoDB!")

def get_database():
    """Get database instance"""
    return mongodb.database

# Helper function to convert ObjectId to string
def serialize_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc

def serialize_docs(docs):
    """Convert list of MongoDB documents to JSON serializable format"""
    return [serialize_doc(doc) for doc in docs]

# main.py - Updated FastAPI application
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from typing import Optional, List
from jose import JWTError, jwt
from datetime import datetime, timedelta
from bson import ObjectId
from contextlib import asynccontextmanager

# Import database functions
from database import connect_to_mongo, close_mongo_connection, get_database, serialize_doc, serialize_docs

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "keeks09Prathi11@143HappyFriends")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()

app = FastAPI(lifespan=lifespan, title="Music App API", version="1.0.0")

# ------------------- ENHANCED MODELS -------------------
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "musiclover"  # Default role

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str

class MusicUpload(BaseModel):
    name: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    duration: Optional[int] = None  # in seconds

class MusicResponse(BaseModel):
    id: str
    name: str
    artist: Optional[str] = None
    genre: Optional[str] = None
    duration: Optional[int] = None
    uploaded_by: str
    uploaded_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

# ------------------- DATABASE OPERATIONS -------------------
class UserService:
    def __init__(self):
        self.db = get_database()
        self.collection = self.db["users"]
    
    async def create_user(self, user_data: dict):
        """Create a new user"""
        result = await self.collection.insert_one(user_data)
        return str(result.inserted_id)
    
    async def get_user_by_username(self, username: str):
        """Get user by username"""
        user = await self.collection.find_one({"username": username})
        return serialize_doc(user) if user else None
    
    async def get_user_by_email(self, email: str):
        """Get user by email"""
        user = await self.collection.find_one({"email": email})
        return serialize_doc(user) if user else None
    
    async def get_all_users(self):
        """Get all users (excluding passwords)"""
        users = []
        async for user in self.collection.find({}, {"password": 0}):
            users.append(serialize_doc(user))
        return users

class MusicService:
    def __init__(self):
        self.db = get_database()
        self.collection = self.db["music"]
    
    async def create_music(self, music_data: dict):
        """Add new music"""
        result = await self.collection.insert_one(music_data)
        return str(result.inserted_id)
    
    async def get_all_music(self):
        """Get all music"""
        music_list = []
        async for music in self.collection.find():
            music_list.append(serialize_doc(music))
        return music_list
    
    async def get_music_by_id(self, music_id: str):
        """Get music by ID"""
        try:
            obj_id = ObjectId(music_id)
            music = await self.collection.find_one({"_id": obj_id})
            return serialize_doc(music) if music else None
        except:
            return None
    
    async def search_music(self, query: str):
        """Search music by name or artist"""
        music_list = []
        search_filter = {
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"artist": {"$regex": query, "$options": "i"}}
            ]
        }
        async for music in self.collection.find(search_filter):
            music_list.append(serialize_doc(music))
        return music_list

# Initialize services
user_service = UserService()
music_service = MusicService()

# ------------------- AUTHENTICATION -------------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await user_service.get_user_by_username(username)
    if not user:
        raise credentials_exception
    return user

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    """Dependency to ensure user is admin"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# ------------------- ROUTES -------------------

@app.post("/register", response_model=dict)
async def register(user: UserRegister):
    # Check if user already exists
    existing_user = await user_service.get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    existing_email = await user_service.get_user_by_email(user.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")

    # Create user
    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
        "role": user.role,
        "created_at": datetime.utcnow()
    }
    
    user_id = await user_service.create_user(user_data)
    return {"msg": "User registered successfully", "user_id": user_id}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Find user by username or email
    user = await user_service.get_user_by_username(form_data.username)
    if not user:
        user = await user_service.get_user_by_email(form_data.username)
    
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["_id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "role": current_user["role"]
    }

# ------------------- USER ROUTES -------------------
@app.get("/users", response_model=List[dict])
async def get_all_users(current_user: dict = Depends(get_admin_user)):
    return await user_service.get_all_users()

@app.get("/users/{username}")
async def get_user(username: str, current_user: dict = Depends(get_current_user)):
    user = await user_service.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Remove password from response
    user.pop("password", None)
    return user

# ------------------- MUSIC ROUTES -------------------
@app.get("/music")
async def get_music(current_user: dict = Depends(get_current_user)):
    return await music_service.get_all_music()

@app.get("/music/{music_id}")
async def get_music_by_id(music_id: str, current_user: dict = Depends(get_current_user)):
    music = await music_service.get_music_by_id(music_id)
    if not music:
        raise HTTPException(status_code=404, detail="Music not found")
    return music

@app.get("/music/search/{query}")
async def search_music(query: str, current_user: dict = Depends(get_current_user)):
    return await music_service.search_music(query)

@app.post("/music")
async def upload_music(
    music: MusicUpload, 
    current_user: dict = Depends(get_admin_user)
):
    music_data = {
        "name": music.name,
        "artist": music.artist,
        "genre": music.genre,
        "duration": music.duration,
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow()
    }
    
    music_id = await music_service.create_music(music_data)
    return {"msg": "Music uploaded successfully", "music_id": music_id}

@app.delete("/music/{music_id}")
async def delete_music(
    music_id: str,
    current_user: dict = Depends(get_admin_user)
):
    db = get_database()
    try:
        obj_id = ObjectId(music_id)
        result = await db["music"].delete_one({"_id": obj_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Music not found")
        
        return {"msg": "Music deleted successfully"}
    except:
        raise HTTPException(status_code=400, detail="Invalid music ID")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)