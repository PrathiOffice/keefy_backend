from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from typing import Optional, List
from jose import JWTError, jwt
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
from dotenv import load_dotenv
import uuid
import aiofiles
import hashlib
from pathlib import Path
from tinytag import TinyTag
from fastapi import FastAPI


app = FastAPI()


# ------------------- CONFIG -------------------
load_dotenv()  # Load environment variables from .env file

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in environment variables")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY must be set in environment variables")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "musicapp")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "app/media"))

# Ensure media directory exists
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
app = FastAPI(
    title="Music App API", 
    version="2.0.0", 
    description="Secure Music Management API with Admin Controls",
    openapi_tags=[
        {"name": "Auth", "description": "User authentication and registration"},
        {"name": "Music", "description": "Music upload and retrieval (Admin upload only)"},
        {"name": "Users", "description": "User management (Admin access for bulk)"}
    ]
)

# MongoDB Init
client = AsyncIOMotorClient(MONGO_URL, uuidRepresentation="standard", serverSelectionTimeoutMS=5000)
db = client[MONGO_DB_NAME]

# Initialize collections
users_col = db["users"]
music_col = db["music"]

# ------------------- MODELS -------------------
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    role: str = Field(..., pattern="^(admin|music_lover)$", description="Role: admin or music_lover")

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class MusicCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Name of the music track")
    artist: Optional[str] = Field(None, max_length=100, description="Artist name")
    genre: Optional[str] = Field(None, max_length=50, description="Music genre")
    description: Optional[str] = Field(None, max_length=500, description="Track description")

class MusicResponse(BaseModel):
    id: str
    name: str
    artist: Optional[str]
    music_director: Optional[str]
    singer: Optional[str]
    genre: Optional[str]
    description: Optional[str]
    uploaded_by: str
    uploaded_at: str
    file_path: Optional[str] = None  # URL or path to audio file
    image: Optional[str] = None  # Path to image

class BulkMusicUploadResponse(BaseModel):
    msg: str
    music_ids: List[str]
    failed: List[dict]
    success_count: int

# ------------------- UTILS -------------------
async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await users_col.find_one({"username": username})
    if not user:
        raise credentials_exception
    return user

async def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user

async def extract_metadata_from_file(temp_file_path: str) -> dict:
    """Extract metadata from audio file using tinytag."""
    try:
        tag = TinyTag.get(temp_file_path)
        metadata = {
            "name": tag.title or "",
            "artist": tag.artist or "",
            "music_director": tag.composer or "",  # Composer often maps to music director
            "singer": tag.artist or "",  # Singer often same as artist; adjust if separate tag available
            "genre": tag.genre or "",
        }
        # Extract front cover image if present
        if tag.images and tag.images.front_cover:
            image_path = MEDIA_DIR / f"{uuid.uuid4()}.jpg"
            with open(image_path, "wb") as img_file:
                img_file.write(tag.images.front_cover.data)
            metadata["image"] = str(image_path.relative_to(MEDIA_DIR))
        return metadata
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return {}

async def save_uploaded_file(upload_file: UploadFile, current_user: dict) -> tuple[str, dict]:
    """Save uploaded audio file, extract metadata, and return file path and metadata."""
    file_extension = Path(upload_file.filename).suffix.lower()
    if file_extension not in ['.mp3', '.wav', '.flac', '.m4a']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only MP3, WAV, FLAC, M4A allowed.")
    
    # Generate unique filename
    file_hash = hashlib.md5(f"{upload_file.filename}{current_user['username']}{datetime.utcnow().timestamp()}".encode()).hexdigest()
    filename = f"{file_hash}{file_extension}"
    file_path = MEDIA_DIR / filename
    
    # Write file content
    async with aiofiles.open(file_path, 'wb') as buffer:
        content = await upload_file.read()
        await buffer.write(content)
    
    # Extract metadata from the saved file
    metadata = await extract_metadata_from_file(str(file_path))
    
    return str(file_path.relative_to(MEDIA_DIR)), metadata

# ------------------- ROUTES -------------------
@app.post("/register", response_model=dict, dependencies=[Depends(verify_api_key)], tags=["Auth"])
async def register(user: UserRegister):
    # Check for existing user
    existing_user = await users_col.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or Email already exists")

    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
        "role": user.role,
        "created_at": datetime.utcnow()
    }
    result = await users_col.insert_one(user_data)
    return {"msg": "User registered successfully", "user_id": str(result.inserted_id)}

@app.post("/login", response_model=Token, dependencies=[Depends(verify_api_key)], tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_col.find_one({"$or": [{"username": form_data.username}, {"email": form_data.username}]})

    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.get("/me", response_model=UserResponse, dependencies=[Depends(verify_api_key)], tags=["Auth"])
async def get_user_data(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user["_id"]),
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
        created_at=current_user["created_at"].isoformat()
    )

# ------------------- USERS -------------------
@app.get("/users", response_model=List[UserResponse], dependencies=[Depends(verify_api_key), Depends(require_admin)], tags=["Users"])
async def get_all_users(skip: int = 0, limit: int = 100):
    users = []
    async for user in users_col.find({}, {"password": 0}).skip(skip).limit(limit).sort("created_at", -1):
        user_dict = dict(user)
        user_dict["id"] = str(user_dict.pop("_id"))
        user_dict["created_at"] = user_dict["created_at"].isoformat()
        users.append(UserResponse(**user_dict))
    return users

@app.get("/users/{username}", response_model=UserResponse, dependencies=[Depends(verify_api_key)], tags=["Users"])
async def get_user(username: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin" and current_user["username"] != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this user")
    
    user = await users_col.find_one({"username": username}, {"password": 0})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    user_dict = dict(user)
    user_dict["id"] = str(user_dict.pop("_id"))
    user_dict["created_at"] = user_dict["created_at"].isoformat()
    return UserResponse(**user_dict)

@app.get("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(verify_api_key), Depends(require_admin)], tags=["Users"])
async def get_user_by_id(user_id: str):
    try:
        obj_id = ObjectId(user_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user ID format")
    
    user = await users_col.find_one({"_id": obj_id}, {"password": 0})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    user_dict = dict(user)
    user_dict["id"] = str(user_dict.pop("_id"))
    user_dict["created_at"] = user_dict["created_at"].isoformat()
    return UserResponse(**user_dict)

# ------------------- MUSIC -------------------
@app.post(
    "/music",
    response_model=dict,
    dependencies=[Depends(verify_api_key), Depends(require_admin)],
    tags=["Music"]
)
async def upload_music(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Save file and extract metadata
    file_path, metadata = await save_uploaded_file(file, current_user)

    # Fallback values for missing metadata
    music_entry = {
        "name": metadata.get("name") or Path(file.filename).stem,  # fallback to filename
        "artist": metadata.get("artist") or "Unknown Artist",
        "music_director": metadata.get("music_director") or None,
        "singer": metadata.get("singer") or None,
        "genre": metadata.get("genre") or None,
        "description": "",
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow(),
        "file_path": file_path,
        "image": metadata.get("image")
    }

    # Find which metadata fields are missing
    missing_metadata = [
        field for field, value in {
            "name": metadata.get("name"),
            "artist": metadata.get("artist"),
            "music_director": metadata.get("music_director"),
            "singer": metadata.get("singer"),
            "genre": metadata.get("genre"),
        }.items() if not value
    ]

    # Save to DB
    result = await music_col.insert_one(music_entry)

    return {
        "msg": "Music uploaded successfully",
        "music_id": str(result.inserted_id),
        "file_path": file_path,
        "missing_metadata": missing_metadata  # tells you what needs manual update
    }



@app.post("/music/bulk", response_model=BulkMusicUploadResponse, dependencies=[Depends(verify_api_key), Depends(require_admin)], tags=["Music"])
async def upload_music_bulk(
    music_list: List[MusicCreate],
    current_user: dict = Depends(get_current_user)
):
    if not music_list:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Music list cannot be empty")
    if len(music_list) > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot upload more than 100 songs at once")

    music_data_list = []
    success_count = 0
    
    for music in music_list:
        music_entry = {
            "name": music.name,
            "artist": music.artist,
            "genre": music.genre,
            "description": music.description,
            "uploaded_by": current_user["username"],
            "uploaded_at": datetime.utcnow()
        }
        music_data_list.append(music_entry)
    
    try:
        result = await music_col.insert_many(music_data_list, ordered=False)
        success_count = len(result.inserted_ids)
        return BulkMusicUploadResponse(
            msg=f"Bulk music upload completed successfully. {success_count} songs uploaded.",
            music_ids=[str(oid) for oid in result.inserted_ids],
            failed=[],
            success_count=success_count
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bulk upload failed: {str(e)}")

@app.get("/music", response_model=List[MusicResponse], dependencies=[Depends(verify_api_key)], tags=["Music"])
async def get_music(skip: int = 0, limit: int = 50, genre: Optional[str] = None):
    query = {}
    if genre:
        query["genre"] = genre
    
    music_list = []
    async for music in music_col.find(query).skip(skip).limit(limit).sort("uploaded_at", -1):
        music_dict = dict(music)
        music_dict["id"] = str(music_dict.pop("_id"))
        music_dict["uploaded_at"] = music_dict["uploaded_at"].isoformat()
        music_list.append(MusicResponse(**music_dict))
    return music_list

@app.get("/music/all", response_model=List[MusicResponse], dependencies=[Depends(verify_api_key)], tags=["Music"])
async def get_all_music():
    """Retrieve all songs in the music collection without pagination."""
    music_list = []
    async for music in music_col.find({}).sort("uploaded_at", -1):
        music_dict = dict(music)
        music_dict["id"] = str(music_dict.pop("_id"))
        music_dict["uploaded_at"] = music_dict["uploaded_at"].isoformat()
        music_list.append(MusicResponse(**music_dict))
    if not music_list:
        return []
    return music_list

@app.get("/music/{music_id}", response_model=MusicResponse, dependencies=[Depends(verify_api_key)], tags=["Music"])
async def get_music_by_id(music_id: str):
    try:
        obj_id = ObjectId(music_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid music ID format")

    music = await music_col.find_one({"_id": obj_id})
    if not music:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Music not found")

    music_dict = dict(music)
    music_dict["id"] = str(music_dict.pop("_id"))
    music_dict["uploaded_at"] = music_dict["uploaded_at"].isoformat()
    return MusicResponse(**music_dict)

@app.get("/music/search", response_model=List[MusicResponse], dependencies=[Depends(verify_api_key)], tags=["Music"])
async def search_music(q: str, limit: int = 20):
    if len(q) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must be at least 2 characters")
    
    query = {
        "$or": [
            {"name": {"$regex": q, "$options": "i"}},
            {"artist": {"$regex": q, "$options": "i"}},
            {"singer": {"$regex": q, "$options": "i"}},
            {"music_director": {"$regex": q, "$options": "i"}},
            {"genre": {"$regex": q, "$options": "i"}}
        ]
    }
    
    music_list = []
    async for music in music_col.find(query).limit(limit).sort("uploaded_at", -1):
        music_dict = dict(music)
        music_dict["id"] = str(music_dict.pop("_id"))
        music_dict["uploaded_at"] = music_dict["uploaded_at"].isoformat()
        music_list.append(MusicResponse(**music_dict))
    return music_list

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "database": MONGO_DB_NAME, "media_dir": str(MEDIA_DIR)}

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    client.close()