from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Query, Form
from fastapi.security import OAuth2PasswordBearer
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
import logging
import re

# ------------------- CONFIG -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
API_KEY = os.getenv("API_KEY")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "musicapp")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "app/media"))
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "app/images"))

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Cached password context for performance
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

app = FastAPI(
    title="Music App API",
    version="2.0.0",
    description="Secure Music Management API with Admin Controls, Master Lists, and Image Support",
    openapi_tags=[
        {"name": "Auth", "description": "User authentication and registration"},
        {"name": "Music", "description": "Music upload and retrieval (Admin upload only)"},
        {"name": "Images", "description": "Image upload and retrieval for music and master data"},
        {"name": "Users", "description": "User management (Admin access for bulk)"},
        {"name": "Master", "description": "Manage master lists for actors, movies, music directors, artists, categories"}
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
images_col = db["images"]

# Initialize master lists and indexes
async def init_db():
    try:
        # Create indexes with error handling
        indexes = [
            (users_col, [("username", 1)], "username_unique_idx", True),
            (users_col, [("email", 1)], "email_unique_idx", True),
            (music_col, [("uploaded_by", 1), ("language", 1), ("category", 1)], "music_search_idx", False),
            (music_col, [("artist", 1), ("music_director", 1)], "music_creators_idx", False),
            (music_col, [("name", 1), ("movie_name", 1), ("actor_name", 1)], "music_names_idx", False),
            (images_col, [("name", 1)], "image_name_unique_idx", True),
            (actors_col, [("name", 1)], "actor_name_idx", True)  # Added index for actor name
        ]
        for col, keys, name, unique in indexes:
            try:
                await col.create_index(keys, unique=unique, name=name)
            except Exception as e:
                if "already exists" in str(e) or "IndexKeySpecsConflict" in str(e):
                    logger.info(f"Index {name} already exists, skipping: {e}")
                else:
                    logger.error(f"Error creating index {name}: {e}")

        # Master data for collections
        master_data = {
            "languages": ["English", "Spanish", "Hindi", "Tamil", "Telugu"],
            "categories": ["Pop", "Love", "Rock", "Classical", "Hip-Hop", "Jazz"],
            "actors": ["Rajinikanth", "Pradeep Ranganathan", "Vijay"],
            "music_directors": ["A.R. Rahman", "John Williams", "Hans Zimmer", "Ilaiyaraaja", "Devi Sri Prasad"],
            "movies": ["Forrest Gump", "Lagaan", "Titanic", "Inception", "LIK"],
            "artists": ["Aamir Khan", "Adele", "Shreya Ghoshal", "Ed Sheeran", "Sagar", "Sumangali"],
            "images": ["default_cover.jpg", "album_art1.jpg", "album_art2.jpg"]
        }
        for collection, items in master_data.items():
            col = await get_collection(collection)
            if col:
                for item in items:
                    try:
                        existing_item = await col.find_one({"name": item})
                        if not existing_item:
                            doc = {"name": item}
                            if collection == "images":
                                doc["file_path"] = f"images/{item}"
                                doc["uploaded_at"] = datetime.utcnow()
                            await col.insert_one(doc)
                    except Exception as e:
                        logger.error(f"Error inserting {item} into {collection}: {e}")
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# ------------------- MODELS -------------------
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern="^(admin|music_lover)$")

class UserInfo(BaseModel):
    username: str
    email: EmailStr
    role: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class ImageCreate(BaseModel):
    actor_name: Optional[str] = None

class MusicCreate(BaseModel):
    name: Optional[str] = None
    artist: Optional[str] = None
    genre: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    category: Optional[str] = None
    is_keefy_original: Optional[bool] = None
    movie_name: Optional[str] = None
    actor_name: Optional[str] = None
    music_director: Optional[str] = None
    singer: Optional[str] = None
    image_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F]{24}$")

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
    image_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F]{24}$")

class MasterDataCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    image_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F]{24}$")

class BulkMasterDataItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    image_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F]{24}$")

class BulkMasterDataCreate(BaseModel):
    items: List[BulkMasterDataItem] = Field(..., min_items=1, max_items=100)

# ------------------- VALIDATORS -------------------
async def validate_master_field(field: str, value: Optional[str], collection):
    if value:
        normalized_value = value.lower()
        existing_item = await collection.find_one({"name": {"$regex": f"^{normalized_value}$", "$options": "i"}})
        if not existing_item:
            raise ValueError(f"Invalid {field}: {value}. Must be one of the predefined {field}s.")
        return value
    return value

async def validate_actor_name(actor_name: Optional[str]):
    if actor_name:
        normalized_value = actor_name.strip().lower()
        logger.info(f"Validating actor_name: {actor_name} (normalized: {normalized_value})")
        actor = await actors_col.find_one({"name": {"$regex": f"^{normalized_value}$", "$options": "i"}})
        if not actor:
            logger.error(f"Actor not found for actor_name: {actor_name}")
            raise ValueError(f"Invalid actor_name: {actor_name}. Must be one of the predefined actors.")
        logger.info(f"Found actor: {actor['name']} with _id: {actor['_id']}")
        return actor_name, actor["_id"]
    return None, None

async def validate_image_id(image_id: Optional[str]):
    if image_id:
        try:
            obj_id = ObjectId(image_id)
            image = await images_col.find_one({"_id": obj_id})
            if not image:
                logger.error(f"Image not found for image_id: {image_id}")
                raise ValueError(f"Invalid image_id: {image_id}")
            logger.info(f"Validated image_id: {image_id}, file_path: {image['file_path']}")
            return image_id, image["file_path"]
        except Exception as e:
            logger.error(f"Invalid image_id format: {image_id}, error: {str(e)}")
            raise ValueError(f"Invalid image_id format: {image_id}")
    return None, None

async def add_movie_if_not_exists(movie_name: Optional[str]):
    if movie_name and movie_name.strip():
        normalized_name = movie_name.lower()
        existing_movie = await movies_col.find_one({"name": {"$regex": f"^{normalized_name}$", "$options": "i"}})
        if not existing_movie:
            await movies_col.insert_one({"name": movie_name})
        return movie_name
    return movie_name

async def get_music_data_from_form(
    name: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    is_keefy_original: Optional[bool] = Form(None),
    movie_name: Optional[str] = Form(None),
    actor_name: Optional[str] = Form(None),
    music_director: Optional[str] = Form(None),
    singer: Optional[str] = Form(None),
    image_id: Optional[str] = Form(None)
) -> MusicCreate:
    music_data = MusicCreate(
        name=name,
        artist=artist,
        genre=genre,
        description=description,
        language=language,
        category=category,
        is_keefy_original=is_keefy_original,
        movie_name=movie_name,
        actor_name=actor_name,
        music_director=music_director,
        singer=singer,
        image_id=image_id
    )
    
    try:
        if music_data.language:
            music_data.language = await validate_master_field("language", music_data.language, languages_col)
        if music_data.category:
            music_data.category = await validate_master_field("category", music_data.category, categories_col)
        if music_data.artist:
            music_data.artist = await validate_master_field("artist", music_data.artist, artists_col)
        if music_data.music_director:
            music_data.music_director = await validate_master_field("music_director", music_data.music_director, music_directors_col)
        if music_data.actor_name:
            music_data.actor_name = await validate_master_field("actor_name", music_data.actor_name, actors_col)
        if music_data.image_id:
            music_data.image_id, _ = await validate_image_id(music_data.image_id)
        music_data.movie_name = await add_movie_if_not_exists(music_data.movie_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return music_data

async def validate_music_update(music: MusicUpdate = Depends()):
    try:
        music.language = await validate_master_field("language", music.language, languages_col)
        music.category = await validate_master_field("category", music.category, categories_col)
        music.artist = await validate_master_field("artist", music.artist, artists_col)
        music.music_director = await validate_master_field("music_director", music.music_director, music_directors_col)
        music.actor_name = await validate_master_field("actor_name", music.actor_name, actors_col)
        music.image_id, _ = await validate_image_id(music.image_id)
        music.movie_name = await add_movie_if_not_exists(music.movie_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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

    logger.info(f"Saving audio file to {file_path}")
    async with aiofiles.open(file_path, 'wb') as buffer:
        while content := await upload_file.read(1024 * 1024):
            await buffer.write(content)

    if not file_path.exists():
        logger.error(f"Failed to save audio file: {file_path}")
        raise HTTPException(status_code=500, detail="Failed to save audio file")

    metadata = await extract_metadata_from_file(str(file_path))
    if metadata.get("movie_name"):
        metadata["movie_name"] = await add_movie_if_not_exists(metadata["movie_name"])
    return str(file_path.relative_to(MEDIA_DIR)), metadata

async def save_uploaded_image(upload_file: UploadFile, current_user: dict) -> tuple[str, str]:
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
    file_extension = Path(upload_file.filename).suffix.lower()
    if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Only {', '.join(ALLOWED_IMAGE_EXTENSIONS)} allowed.")

    file_hash = hashlib.md5(f"{upload_file.filename}{current_user['username']}{datetime.utcnow().timestamp()}".encode()).hexdigest()
    filename = f"{file_hash}{file_extension}"
    file_path = IMAGE_DIR / filename

    logger.info(f"Saving image to {file_path}")
    async with aiofiles.open(file_path, 'wb') as buffer:
        while content := await upload_file.read(1024 * 1024):
            await buffer.write(content)

    if not file_path.exists():
        logger.error(f"Failed to save image file: {file_path}")
        raise HTTPException(status_code=500, detail="Failed to save image file")

    image_entry = {
        "name": upload_file.filename,
        "file_path": str(file_path.relative_to(IMAGE_DIR)),
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow()
    }
    result = await images_col.insert_one(image_entry)
    return str(file_path.relative_to(IMAGE_DIR)), str(result.inserted_id)

async def get_collection(collection_name: str):
    collections = {
        "actors": actors_col,
        "movies": movies_col,
        "music_directors": music_directors_col,
        "artists": artists_col,
        "categories": categories_col,
        "languages": languages_col,
        "images": images_col
    }
    return collections.get(collection_name)

async def get_collection_all():
    return {
        "actors": actors_col,
        "movies": movies_col,
        "music_directors": music_directors_col,
        "artists": artists_col,
        "categories": categories_col,
        "languages": languages_col,
        "images": images_col
    }

async def fix_actor_image_ids():
    actors = await actors_col.find().to_list(length=1000)
    for actor in actors:
        image_id = actor.get("image_id")
        if image_id:
            try:
                obj_id = ObjectId(image_id)
                image = await images_col.find_one({"_id": obj_id})
                if not image:
                    logger.warning(f"Invalid image_id {image_id} for actor {actor['name']}, removing reference")
                    await actors_col.update_one(
                        {"_id": actor["_id"]},
                        {"$unset": {"image_id": "", "image_path": ""}}
                    )
                elif image["file_path"] != actor.get("image_path"):
                    logger.info(f"Fixing image_path for actor {actor['name']}")
                    await actors_col.update_one(
                        {"_id": actor["_id"]},
                        {"$set": {"image_path": image["file_path"]}}
                    )
            except Exception:
                logger.error(f"Invalid image_id format {image_id} for actor {actor['name']}")
                await actors_col.update_one(
                    {"_id": actor["_id"]},
                    {"$unset": {"image_id": "", "image_path": ""}}
                )

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
async def login(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(..., pattern="^(admin|music_lover)$")
):
    user = await users_col.find_one({"$or": [{"username": username}, {"email": username}]})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    if not verify_password(password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    if role != user["role"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid role: You are not a {role}")
    
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": access_token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}

# ------------------- USER INFO -------------------
@app.get("/me", response_model=UserInfo, tags=["Users"], dependencies=[Depends(verify_api_key)])
async def get_user_info(current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    return UserInfo(
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
        created_at=current_user["created_at"]
    )

# ------------------- MASTER LIST MANAGEMENT -------------------
@app.get("/master/{collection}", response_model=List[dict], tags=["Master"], dependencies=[Depends(verify_api_key)])
async def get_master_list(collection: str):
    col = await get_collection(collection)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    items = await col.find().to_list(length=1000)
    return [
        {
            "name": item["name"],
            "image_id": str(item.get("image_id")) if item.get("image_id") else None,
            "image_path": item.get("image_path")
        }
        for item in items
    ]

@app.get("/master/search", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key)])
async def search_master(q: Optional[str] = Query(None), collections: Optional[List[str]] = Query(None)):
    all_collections = await get_collection_all()
    target_collections = {k: v for k, v in all_collections.items() if not collections or k in collections}
    if not target_collections:
        raise HTTPException(status_code=400, detail=f"Invalid collection(s)")
    results = {}
    for name, col in target_collections.items():
        matches = await col.find({"name": {"$regex": q, "$options": "i"}} if q else {}).to_list(length=1000)
        results[name] = [
            {
                "name": doc["name"],
                "image_id": str(doc.get("image_id")) if doc.get("image_id") else None,
                "image_path": doc.get("image_path")
            }
            for doc in matches
        ]
    return {"query": q, "results": results} if q else {"results": results}

@app.post("/master/{collection}", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def add_master_data(
    collection: str,
    name: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    image_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    col = await get_collection(collection)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    
    normalized_name = name.lower()
    existing_item = await col.find_one({"name": {"$regex": f"^{normalized_name}$", "$options": "i"}})
    if existing_item:
        raise HTTPException(status_code=400, detail=f"Item '{name}' already exists")
    
    master_data = {"name": name}
    if image_file:
        image_path, image_id = await save_uploaded_image(image_file, current_user)
        master_data["image_id"] = ObjectId(image_id)
        master_data["image_path"] = image_path
    elif image_id:
        validated_image_id, image_path = await validate_image_id(image_id)
        if not validated_image_id:
            raise HTTPException(status_code=400, detail=f"Image with ID {image_id} not found")
        master_data["image_id"] = ObjectId(validated_image_id)
        master_data["image_path"] = image_path
    
    logger.info(f"Storing master data for {collection}: name={name}, image_id={master_data.get('image_id')}, image_path={master_data.get('image_path')}")
    result = await col.insert_one(master_data)
    return {"msg": f"Added '{name}' to {collection}", "item_id": str(result.inserted_id), "success": True}

@app.post("/master/{collection}/bulk", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def add_bulk_master_data(collection: str, bulk_data: BulkMasterDataCreate):
    col = await get_collection(collection)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    
    inserted_count = 0
    failed_items = []
    existing_items = []
    
    for item in bulk_data.items:
        if not item.name.strip():
            failed_items.append({"item": item.name, "reason": "Empty name"})
            continue
        normalized_item = item.name.lower()
        existing = await col.find_one({"name": {"$regex": f"^{normalized_item}$", "$options": "i"}})
        if existing:
            existing_items.append(item.name)
            continue
        try:
            master_data = {"name": item.name}
            if item.image_id:
                image_id, image_path = await validate_image_id(item.image_id)
                if not image_id:
                    failed_items.append({"item": item.name, "reason": f"Image with ID {item.image_id} not found"})
                    continue
                master_data["image_id"] = ObjectId(image_id)
                master_data["image_path"] = image_path
            logger.info(f"Storing bulk master data for {collection}: name={item.name}, image_id={master_data.get('image_id')}, image_path={master_data.get('image_path')}")
            await col.insert_one(master_data)
            inserted_count += 1
        except Exception as e:
            failed_items.append({"item": item.name, "reason": str(e)})
    
    response = {
        "msg": f"Processed bulk insert for {collection}",
        "inserted_count": inserted_count,
        "total_submitted": len(bulk_data.items),
        "success": inserted_count > 0
    }
    if existing_items:
        response["existing_items"] = existing_items
    if failed_items:
        response["failed_items"] = failed_items
    return response

# ------------------- MUSIC -------------------
@app.post("/music", response_model=dict, tags=["Music"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def upload_music(
    file: UploadFile = File(...),
    music_data: MusicCreate = Depends(get_music_data_from_form),
    image_file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    file_path, metadata = await save_uploaded_file(file, current_user)
    
    image_id = music_data.image_id
    image_path = None
    if image_file:
        image_path, image_id = await save_uploaded_image(image_file, current_user)
    elif music_data.image_id:
        image_id, image_path = await validate_image_id(music_data.image_id)
        if not image_id:
            raise HTTPException(status_code=400, detail=f"Image with ID {music_data.image_id} not found")

    music_entry = {
        "name": music_data.name or metadata.get("name") or f"Unnamed_{file.filename}",
        "artist": music_data.artist or metadata.get("artist") or "",
        "music_director": music_data.music_director or metadata.get("music_director") or "",
        "singer": music_data.singer or metadata.get("singer") or "",
        "genre": music_data.genre or metadata.get("genre") or "",
        "description": music_data.description or "",
        "language": music_data.language or metadata.get("language") or "",
        "category": music_data.category or metadata.get("category") or "",
        "is_keefy_original": music_data.is_keefy_original if music_data.is_keefy_original is not None else False,
        "movie_name": music_data.movie_name or metadata.get("movie_name") or "",
        "actor_name": music_data.actor_name or "",
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow(),
        "file_path": file_path,
        "image_id": ObjectId(image_id) if image_id else None,
        "image_path": image_path,
        "is_completed": False
    }
    
    result = await music_col.insert_one(music_entry)
    stored_data = {k: str(v) if isinstance(v, ObjectId) else v for k, v in music_entry.items()}
    
    return {
        "msg": "Music uploaded successfully",
        "music_id": str(result.inserted_id),
        "file_path": file_path,
        "image_id": image_id,
        "image_path": image_path,
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
    match_conditions = [
        {field: {"$regex": value, "$options": "i"}}
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
        ]
        if value
    ]
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
            "file_path": 1, "is_completed": 1,
            "image_id": {"$toString": "$image_id"},
            "image_path": 1
        }}
    ]
    return await music_col.aggregate(pipeline).to_list(length=limit)

@app.put("/music/{music_id}", response_model=dict, tags=["Music"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def update_music_metadata(music_id: str, music_data: MusicUpdate = Depends(validate_music_update)):
    try:
        obj_id = ObjectId(music_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid music ID format")
    
    update_data = music_data.dict(exclude_unset=True)
    update_data["is_completed"] = True
    if music_data.image_id:
        image_id, image_path = await validate_image_id(music_data.image_id)
        if image_id:
            update_data["image_id"] = ObjectId(image_id)
            update_data["image_path"] = image_path
        else:
            raise HTTPException(status_code=400, detail=f"Image with ID {music_data.image_id} not found")
    
    result = await music_col.update_one({"_id": obj_id}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Music not found or no changes applied")
    return {"msg": "Music metadata updated successfully", "music_id": music_id, "success": True}

@app.get("/music/play/{music_id}", response_model=None, tags=["Music"], dependencies=[Depends(verify_api_key)])
async def play_music(music_id: str):
    try:
        obj_id = ObjectId(music_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid music ID format")
    
    music = await music_col.find_one({"_id": obj_id})
    if not music:
        raise HTTPException(status_code=404, detail="Music not found")
    file_path = MEDIA_DIR / music["file_path"]
    if not file_path.exists():
        logger.error(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="audio/mpeg", filename=file_path.name)

# ------------------- IMAGES -------------------
@app.post("/images", response_model=dict, tags=["Images"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def upload_image(
    file: UploadFile = File(...),
    actor_name: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    # Log the incoming request
    logger.info(f"Received image upload request with actor_name: {actor_name}")
    
    # Validate actor_name if provided
    actor_name_validated, actor_id = await validate_actor_name(actor_name)
    
    # Save the uploaded image
    file_path, image_id = await save_uploaded_image(file, current_user)
    
    # If actor_name is provided, update the actor's image_id and image_path
    if actor_name_validated and actor_id:
        try:
            result = await actors_col.update_one(
                {"_id": actor_id},
                {"$set": {"image_id": ObjectId(image_id), "image_path": file_path}}
            )
            if result.modified_count == 0:
                logger.warning(f"No actor found or updated for actor_name: {actor_name_validated}, actor_id: {actor_id}")
                raise HTTPException(status_code=404, detail=f"Failed to update actor {actor_name_validated}: Actor not found or no changes applied")
            else:
                logger.info(f"Successfully updated actor {actor_name_validated} (ID: {actor_id}) with image_id: {image_id}, image_path: {file_path}")
        except Exception as e:
            logger.error(f"Failed to update actor {actor_name_validated} with image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to update actor with image: {str(e)}")

    return {
        "msg": "Image uploaded successfully",
        "image_id": image_id,
        "file_path": file_path,
        "actor_name": actor_name_validated,
        "success": True
    }

@app.get("/images", response_model=List[dict], tags=["Images"], dependencies=[Depends(verify_api_key)])
async def get_all_images(skip: int = 0, limit: int = 100):
    pipeline = [
        {"$sort": {"uploaded_at": -1}},
        {"$skip": skip},
        {"$limit": limit},
        {"$project": {
            "_id": {"$toString": "$_id"},
            "name": 1,
            "file_path": 1,
            "uploaded_by": 1,
            "uploaded_at": {"$dateToString": {"format": "%Y-%m-%d %H:%M:%S", "date": "$uploaded_at"}}
        }}
    ]
    return await images_col.aggregate(pipeline).to_list(length=limit)

@app.get("/images/{image_id}", response_model=None, tags=["Images"], dependencies=[Depends(verify_api_key)])
async def get_image(image_id: str):
    try:
        obj_id = ObjectId(image_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image ID format")
    
    image = await images_col.find_one({"_id": obj_id})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    file_path = IMAGE_DIR / image["file_path"]
    logger.info(f"Attempting to retrieve image: {file_path}")
    if not file_path.exists():
        logger.error(f"Image file not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"Image file not found: {file_path}")
    return FileResponse(path=file_path, media_type="image/jpeg", filename=image["name"])

# ------------------- HEALTH -------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "database": MONGO_DB_NAME,
        "media_dir": str(MEDIA_DIR),
        "image_dir": str(IMAGE_DIR)
    }

@app.on_event("startup")
async def startup_event():
    await init_db()
    await fix_actor_image_ids()

@app.on_event("shutdown")
async def shutdown_event():
    client.close()