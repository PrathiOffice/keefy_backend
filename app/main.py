from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Query, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from typing import Optional, List, Annotated
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
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

app = FastAPI(
    title="Music App API",
    version="2.0.0",
    description="Secure Music Management API with Admin Controls and Master Lists",
    openapi_tags=[
        {"name": "Auth", "description": "User authentication and registration"},
        {"name": "Music", "description": "Music upload and retrieval (Admin upload only)"},
        {"name": "Users", "description": "User management (Admin access for bulk)"},
        {"name": "Master", "description": "Retrieve master lists for actors, movies, music directors, artists"}
    ]
)

# ------------------- DATABASE -------------------
client = AsyncIOMotorClient(
    MONGO_URL,
    uuidRepresentation="standard",
    serverSelectionTimeoutMS=5000,
    maxPoolSize=50,
    minPoolSize=10
)
db = client[MONGO_DB_NAME]
users_col = db["users"]
music_col = db["music"]
languages_col = db["languages"]
categories_col = db["categories"]
actors_col = db["actors"]
music_directors_col = db["music_directors"]
movies_col = db["movies"]
artists_col = db["artists"]

# Initialize master lists and indexes
async def init_db():
    try:
        # Create indexes with error handling
        await users_col.create_index([("username", 1)], unique=True, partialFilterExpression={"username": {"$exists": True}})
        await users_col.create_index([("email", 1)], unique=True, partialFilterExpression={"email": {"$exists": True}})
        await music_col.create_index([("uploaded_by", 1), ("language", 1), ("category", 1)])
        await music_col.create_index([("artist", 1), ("music_director", 1)])
        await music_col.create_index([("name", 1), ("movie_name", 1), ("actor_name", 1)])
        await artists_col.create_index([("name", 1)], unique=True, partialFilterExpression={"name": {"$exists": True}})

        # Master data for collections
        master_data = {
            "languages": ["English", "Spanish", "Hindi", "Tamil", "Telugu"],
            "categories": ["Pop", "Love", "Rock", "Classical", "Hip-Hop", "Jazz"],
            "actors": ["Tom Hanks", "Aamir Khan", "Priyanka Chopra", "Leonardo DiCaprio", "Rajinikanth"],
            "music_directors": ["A.R. Rahman", "John Williams", "Hans Zimmer", "Ilaiyaraaja"],
            "movies": ["Forrest Gump", "Lagaan", "Titanic", "Inception"],
            "artists": ["Aamir Khan", "Adele", "Shreya Ghoshal", "Ed Sheeran"]
        }

        # Insert master data only if it doesn't exist
        for collection, items in master_data.items():
            col = await get_collection(collection)
            for item in items:
                try:
                    existing_item = await col.find_one({"name": item})
                    if not existing_item:
                        await col.insert_one({"name": item})
                except Exception as e:
                    print(f"Error inserting {item} into {collection}: {e}")
                    # Continue to avoid crashing on duplicate or minor errors
    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize database")
# ------------------- MODELS -------------------
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern="^(admin|music_lover)$")

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class MusicCreate(BaseModel):
    name: Annotated[Optional[str], Form(None)] = None
    artist: Annotated[Optional[str], Form(None)] = None
    genre: Annotated[Optional[str], Form(None)] = None
    description: Annotated[Optional[str], Form(None)] = None
    language: Annotated[Optional[str], Form(None)] = None
    category: Annotated[Optional[str], Form(None)] = None
    is_keefy_original: Annotated[Optional[bool], Form(None)] = None
    movie_name: Annotated[Optional[str], Form(None)] = None
    actor_name: Annotated[Optional[str], Form(None)] = None
    music_director: Annotated[Optional[str], Form(None)] = None
    singer: Annotated[Optional[str], Form(None)] = None

class MusicUpdate(BaseModel):
    name: str = Field(..., min_length=1)
    artist: str = Field(..., min_length=1)
    music_director: Optional[str] = None
    singer: Optional[str] = None
    genre: str = Field(..., min_length=1)
    description: Optional[str] = None
    language: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    is_keefy_original: bool
    movie_name: Optional[str] = None
    actor_name: Optional[str] = None

# ------------------- VALIDATORS -------------------
async def validate_master_field(field: str, value: Optional[str], collection):
    if value and not await collection.find_one({"name": value}):
        raise ValueError(f"Invalid {field}: {value}. Must be one of the predefined {field}s.")
    return value

async def validate_music_create(music: MusicCreate = Depends()):
    if music.language:
        music.language = await validate_master_field("language", music.language, languages_col)
    if music.category:
        music.category = await validate_master_field("category", music.category, categories_col)
    if music.artist:
        music.artist = await validate_master_field("artist", music.artist, artists_col)
    if music.music_director:
        music.music_director = await validate_master_field("music_director", music.music_director, music_directors_col)
    if music.movie_name:
        music.movie_name = await validate_master_field("movie_name", music.movie_name, movies_col)
    if music.actor_name:
        music.actor_name = await validate_master_field("actor_name", music.actor_name, actors_col)
    return music

async def validate_music_update(music: MusicUpdate = Depends()):
    music.language = await validate_master_field("language", music.language, languages_col)
    music.category = await validate_master_field("category", music.category, categories_col)
    music.artist = await validate_master_field("artist", music.artist, artists_col)
    music.music_director = await validate_master_field("music_director", music.music_director, music_directors_col)
    music.movie_name = await validate_master_field("movie_name", music.movie_name, movies_col)
    music.actor_name = await validate_master_field("actor_name", music.actor_name, actors_col)
    return music

# ------------------- UTILS -------------------
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    user = await users_col.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

async def require_admin(current_user: dict = Depends(get_current_user)):
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user

async def extract_metadata_from_file(temp_file_path: str) -> dict:
    try:
        tag = TinyTag.get(temp_file_path)
        return {
            "name": tag.title or "",
            "artist": tag.artist or "",
            "music_director": tag.composer or "",
            "singer": tag.artist or "",
            "genre": tag.genre or "",
            "movie_name": tag.album or ""
        }
    except Exception:
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
        while content := await upload_file.read(1024 * 1024):
            await buffer.write(content)

    metadata = await extract_metadata_from_file(str(file_path))
    return str(file_path.relative_to(MEDIA_DIR)), metadata

async def get_collection(collection_name: str):
    collections = {
        "actors": actors_col,
        "movies": movies_col,
        "music_directors": music_directors_col,
        "artists": artists_col,
        "categories": categories_col,
        "languages": languages_col
    }
    return collections.get(collection_name)

async def get_collection_all():
    return {
        "actors": actors_col,
        "movies": movies_col,
        "music_directors": music_directors_col,
        "artists": artists_col,
        "categories": categories_col,
        "languages": languages_col
    }

# ------------------- AUTH -------------------
@app.post("/register", response_model=dict, tags=["Auth"], dependencies=[Depends(verify_api_key)])
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

@app.post("/login", response_model=Token, tags=["Auth"], dependencies=[Depends(verify_api_key)])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_col.find_one({"$or": [{"username": form_data.username}, {"email": form_data.username}]})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}

# ------------------- MASTER LIST MANAGEMENT -------------------
@app.get("/master/{collection}", response_model=List[str], tags=["Master"], dependencies=[Depends(verify_api_key)])
async def get_master_list(collection: str):
    col = await get_collection(collection)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    items = await col.find().to_list(length=1000)
    return [item["name"] for item in items]

@app.get("/master/search", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key)])
async def search_master(q: Optional[str] = Query(None), collections: Optional[List[str]] = Query(None)):
    all_collections = await get_collection_all()
    target_collections = {k: v for k, v in all_collections.items() if not collections or k in collections}
    if not target_collections:
        raise HTTPException(status_code=400, detail=f"Invalid collection(s). Must be one or more of {list(all_collections.keys())}")
    results = {}
    for name, col in target_collections.items():
        if q:
            matches = await col.find({"name": {"$regex": q, "$options": "i"}}).to_list(length=20)
        else:
            matches = await col.find().to_list(length=1000)
        results[name] = [doc["name"] for doc in matches]
    return {"query": q, "results": results} if q else {"results": results}

# ------------------- MUSIC -------------------
@app.post("/music", response_model=dict, tags=["Music"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def upload_music(file: UploadFile = File(...), music_data: MusicCreate = Depends(validate_music_create), current_user: dict = Depends(get_current_user)):
    file_path, metadata = await save_uploaded_file(file, current_user)
    music_entry = {
        "name": music_data.name or metadata.get("name", f"Unnamed_{file.filename}"),
        "artist": music_data.artist or metadata.get("artist", ""),
        "music_director": music_data.music_director or metadata.get("music_director", ""),
        "singer": music_data.singer or metadata.get("singer", ""),
        "genre": music_data.genre or metadata.get("genre", ""),
        "description": music_data.description or "",
        "language": music_data.language or metadata.get("language", ""),
        "category": music_data.category or metadata.get("category", ""),
        "is_keefy_original": music_data.is_keefy_original if music_data.is_keefy_original is not None else False,
        "movie_name": music_data.movie_name or metadata.get("movie_name", ""),
        "actor_name": music_data.actor_name or "",
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow(),
        "file_path": file_path,
        "is_completed": False
    }
    result = await music_col.insert_one(music_entry)
    stored_data = {k: str(v) if isinstance(v, ObjectId) else v for k, v in music_entry.items()}
    return {
        "msg": "Music uploaded successfully",
        "music_id": str(result.inserted_id),
        "file_path": file_path,
        "provided_data": music_data.dict(exclude_unset=True),
        "stored_data": stored_data,
        "success": True
    }

@app.get("/music", response_model=List[dict], tags=["Music"], dependencies=[Depends(verify_api_key)])
async def get_all_music(
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    artist: Optional[str] = None,
    music_director: Optional[str] = None,
    singer: Optional[str] = None,
    genre: Optional[str] = None,
    description: Optional[str] = None,
    language: Optional[str] = None,
    category: Optional[str] = None,
    is_keefy_original: Optional[bool] = None,
    movie_name: Optional[str] = None,
    actor_name: Optional[str] = None
):
    match_conditions = []
    for field, value in [
        ("name", name),
        ("artist", artist),
        ("music_director", music_director),
        ("singer", singer),
        ("genre", genre),
        ("description", description),
        ("language", language),
        ("category", category),
        ("movie_name", movie_name),
        ("actor_name", actor_name)
    ]:
        if value:
            if field in ["is_keefy_original"]:
                match_conditions.append({field: value})
            else:
                match_conditions.append({field: {"$regex": value, "$options": "i"}})

    # Add is_keefy_original separately if provided, as it's a boolean
    if is_keefy_original is not None:
        match_conditions.append({"is_keefy_original": is_keefy_original})

    pipeline = (
        [{"$match": {"$and": match_conditions}}] if match_conditions else []
    ) + [
        {"$sort": {"uploaded_at": -1}},
        {"$skip": skip},
        {"$limit": limit},
        {"$project": {
            "_id": {"$toString": "$_id"},
            "name": 1, "artist": 1, "music_director": 1, "singer": 1,
            "genre": 1, "description": 1, "language": 1, "category": 1,
            "is_keefy_original": 1, "movie_name": 1, "actor_name": 1,
            "uploaded_by": 1,
            "uploaded_at": {"$dateToString": {"format": "%Y-%m-%d %H:%M:%S", "date": "$uploaded_at"}},
            "file_path": 1, "is_completed": 1
        }}
    ]
    return await music_col.aggregate(pipeline).to_list(length=limit)


@app.put("/music/{music_id}", response_model=dict, tags=["Music"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def update_music_metadata(music_id: str, music_data: MusicUpdate = Depends(validate_music_update)):
    obj_id = ObjectId(music_id)
    update_data = music_data.dict(exclude_unset=True)
    update_data["is_completed"] = True
    result = await music_col.update_one({"_id": obj_id}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Music not found or no changes applied")
    return {"msg": "Music metadata updated successfully", "music_id": music_id, "success": True}

@app.get("/music/play/{music_id}", response_model=None, tags=["Music"], dependencies=[Depends(verify_api_key)])
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
    return {
        "status": "healthy",
        "database": MONGO_DB_NAME,
        "media_dir": str(MEDIA_DIR)
    }



@app.on_event("shutdown")
async def shutdown_event():
    client.close()