from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
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
from typing import List

# ------------------- CONFIG -------------------
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
API_KEY = os.getenv("API_KEY")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "musicapp")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "app/media"))

MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Cached password context for performance
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

# ------------------- DATABASE -------------------
client = AsyncIOMotorClient(
    MONGO_URL,
    uuidRepresentation="standard",
    serverSelectionTimeoutMS=5000,
    maxPoolSize=50,  # Connection pooling
    minPoolSize=10
)
db = client[MONGO_DB_NAME]
users_col = db["users"]
music_col = db["music"]

# Create indexes for faster queries
async def init_db():
    await users_col.create_index([("username", 1)], unique=True)
    await users_col.create_index([("email", 1)], unique=True)
    await music_col.create_index([("uploaded_by", 1)])

@app.on_event("startup")
async def startup_event():
    await init_db()

# ------------------- MODELS -------------------
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern="^(admin|music_lover)$")

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
    name: str = Field(..., min_length=1)
    artist: Optional[str]
    genre: Optional[str]
    description: Optional[str]
    language: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    is_keefy_original: bool = Field(...)

class MusicUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    artist: str = Field(..., min_length=1)
    music_director: Optional[str] = None
    singer: Optional[str] = None
    genre: str = Field(..., min_length=1)
    description: Optional[str] = None
    language: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    is_keefy_original: bool = Field(...)

# ------------------- UTILS -------------------
async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})
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
        return {
            "name": tag.title or "",
            "artist": tag.artist or "",
            "music_director": tag.composer or "",
            "singer": tag.artist or "",
            "genre": tag.genre or "",
        }
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return {}

async def save_uploaded_file(upload_file: UploadFile, current_user: dict) -> tuple[str, dict]:
    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
    file_extension = Path(upload_file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Only {', '.join(ALLOWED_EXTENSIONS)} allowed.")

    file_hash = hashlib.md5(f"{upload_file.filename}{current_user['username']}{datetime.utcnow().timestamp()}".encode()).hexdigest()
    filename = f"{file_hash}{file_extension}"
    file_path = MEDIA_DIR / filename

    async with aiofiles.open(file_path, 'wb') as buffer:
        while content := await upload_file.read(1024 * 1024):  # Read in 1MB chunks
            await buffer.write(content)

    metadata = await extract_metadata_from_file(str(file_path))
    return str(file_path.relative_to(MEDIA_DIR)), metadata

# ------------------- AUTH -------------------
@app.post("/register", response_model=dict, dependencies=[Depends(verify_api_key)])
async def register(user: UserRegister):
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

@app.post("/login", response_model=Token, dependencies=[Depends(verify_api_key)])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_col.find_one({"$or": [{"username": form_data.username}, {"email": form_data.username}]})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}

# ------------------- MUSIC -------------------
@app.post("/music", response_model=dict, dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def upload_music(
    file: UploadFile = File(...),
    music_data: MusicCreate = Depends(),
    current_user: dict = Depends(get_current_user)
):
    file_path, metadata = await save_uploaded_file(file, current_user)

    music_entry = {
        "name": music_data.name or metadata.get("name"),
        "artist": music_data.artist or metadata.get("artist"),
        "music_director": metadata.get("music_director"),
        "singer": metadata.get("singer"),
        "genre": music_data.genre or metadata.get("genre"),
        "description": music_data.description or "",
        "language": music_data.language,
        "category": music_data.category,
        "is_keefy_original": music_data.is_keefy_original,
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow(),
        "file_path": file_path,
        "is_completed": False
    }

    required_fields = ["name", "artist", "genre", "language", "category", "is_keefy_original"]
    missing_metadata = [field for field in required_fields if not music_entry.get(field)]

    if missing_metadata:
        return {
            "msg": "File uploaded. Metadata incomplete.",
            "music_id": str(await music_col.insert_one(music_entry).inserted_id),
            "file_path": file_path,
            "missing_metadata": missing_metadata,
            "success": False
        }

    result = await music_col.insert_one(music_entry)
    return {
        "msg": "Music uploaded successfully",
        "music_id": str(result.inserted_id),
        "file_path": file_path,
        "success": True
    }


@app.get("/music", response_model=List[dict], tags=["Music"])
async def get_all_music(
    skip: int = 0,
    limit: int = 100,  # Optional pagination
    current_user: Optional[dict] = Depends(get_current_user, use_cache=False)  # Optional: for authenticated users only if needed
):
    """
    Retrieve all music entries.
    - skip: Number of records to skip (for pagination).
    - limit: Max number of records to return.
    """
    # If you want admin-only, add: dependencies=[Depends(require_admin)]
    pipeline = [
        {"$sort": {"uploaded_at": -1}},  # Newest first
        {"$skip": skip},
        {"$limit": limit},
        {"$project": {
            "_id": {"$toString": "$_id"},
            "name": 1,
            "artist": 1,
            "music_director": 1,
            "singer": 1,
            "genre": 1,
            "description": 1,
            "language": 1,
            "category": 1,
            "is_keefy_original": 1,
            "uploaded_by": 1,
            "uploaded_at": {"$dateToString": {"format": "%Y-%m-%d %H:%M:%S", "date": "$uploaded_at"}},
            "file_path": 1,
            "is_completed": 1
        }}
    ]
    songs = await music_col.aggregate(pipeline).to_list(length=limit)
    return songs

@app.put("/music/{music_id}", response_model=dict, dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def update_music_metadata(music_id: str, music_data: MusicUpdate, current_user: dict = Depends(get_current_user)):
    obj_id = ObjectId(music_id)
    update_data = music_data.dict(exclude_unset=True)
    update_data["is_completed"] = True
    result = await music_col.update_one({"_id": obj_id}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Music not found or no changes applied")
    return {"msg": "Music metadata updated successfully", "music_id": music_id, "success": True}

@app.get("/music/play/{music_id}", tags=["Music"])
async def play_music(music_id: str):
    obj_id = ObjectId(music_id)
    music = await music_col.find_one({"_id": obj_id})
    if not music:
        raise HTTPException(status_code=404, detail="Music not found")

    file_path = MEDIA_DIR / music["file_path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, media_type="audio/mpeg", filename=file_path.name)

# ------------------- HEALTH -------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "database": MONGO_DB_NAME, "media_dir": str(MEDIA_DIR)}

@app.on_event("shutdown")
async def shutdown_event():
    client.close()