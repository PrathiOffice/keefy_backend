from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Query, Form
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
        # Create indexes with error handling - use specific names to avoid conflicts
        try:
            await users_col.create_index([("username", 1)], unique=True, name="username_unique_idx")
        except Exception as e:
            if "already exists" in str(e) or "IndexKeySpecsConflict" in str(e):
                print(f"Username index already exists, skipping: {e}")
            else:
                print(f"Error creating username index: {e}")
        
        try:
            await users_col.create_index([("email", 1)], unique=True, name="email_unique_idx")
        except Exception as e:
            if "already exists" in str(e) or "IndexKeySpecsConflict" in str(e):
                print(f"Email index already exists, skipping: {e}")
            else:
                print(f"Error creating email index: {e}")
        
        # Create other indexes with error handling
        indexes_to_create = [
            (music_col, [("uploaded_by", 1), ("language", 1), ("category", 1)], "music_search_idx"),
            (music_col, [("artist", 1), ("music_director", 1)], "music_creators_idx"),
            (music_col, [("name", 1), ("movie_name", 1), ("actor_name", 1)], "music_names_idx"),
        ]
        
        for col, keys, name in indexes_to_create:
            try:
                await col.create_index(keys, name=name)
            except Exception as e:
                if "already exists" in str(e) or "IndexKeySpecsConflict" in str(e):
                    print(f"Index {name} already exists, skipping: {e}")
                else:
                    print(f"Error creating index {name}: {e}")

        # Create unique indexes for master collections
        master_collections = ["actors", "movies", "music_directors", "artists", "categories", "languages"]
        for collection_name in master_collections:
            try:
                col = await get_collection(collection_name)
                if col:
                    await col.create_index([("name", 1)], unique=True, name=f"{collection_name}_name_unique_idx")
            except Exception as e:
                if "already exists" in str(e) or "IndexKeySpecsConflict" in str(e):
                    print(f"Index for {collection_name} already exists, skipping: {e}")
                else:
                    print(f"Error creating index for {collection_name}: {e}")

        # Master data for collections
        master_data = {
            "languages": ["English", "Spanish", "Hindi", "Tamil", "Telugu"],
            "categories": ["Pop", "Love", "Rock", "Classical", "Hip-Hop", "Jazz"],
"actors": [
        # Tamil Actors
        "Pradeep Ranganathan", "Ajith", "Dhanush", "Vijay", "Rajinikanth", 
        "Sivakarthikeyan", "Suriya", "Karthi", "Vikram", "Jayam Ravi",
        "Arya", "Vishal", "Jiiva", "Santhanam", "Soori",
        
        # Tamil Actresses
        "Nayanthara", "Trisha", "Samantha", "Shruti Haasan", "Hansika Motwani",
        "Keerthy Suresh", "Anushka Shetty", "Tamannaah", "Kajal Aggarwal",
        
        # Hindi/Bollywood Actors
        "Shah Rukh Khan", "Aamir Khan", "Salman Khan", "Akshay Kumar", 
        "Hrithik Roshan", "Ranbir Kapoor", "Ranveer Singh", "Varun Dhawan",
        "Ayushmann Khurrana", "Rajkummar Rao",
        
        # Hindi/Bollywood Actresses
        "Deepika Padukone", "Priyanka Chopra", "Kareena Kapoor", "Katrina Kaif",
        "Alia Bhatt", "Anushka Sharma", "Sonam Kapoor",
        
        # Hollywood Actors
        "Tom Hanks", "Leonardo DiCaprio", "Brad Pitt", "Will Smith",
        "Robert Downey Jr.", "Chris Evans", "Dwayne Johnson"
    ],           
    "music_directors": [
        "A.R. Rahman", "Ilaiyaraaja", "Devi Sri Prasad", "Anirudh Ravichander",
        "Yuvan Shankar Raja", "Harris Jayaraj", "Thaman", "G.V. Prakash Kumar",
        "Santhosh Narayanan", "Imman", "John Williams", "Hans Zimmer", 
        "Alan Silvestri", "Thomas Newman"
    ],            
    "movies": ["Forrest Gump", "Lagaan", "Titanic", "Inception", "LIK"],
            
    "artists": ["Aamir Khan", "Adele", "Shreya Ghoshal", "Ed Sheeran", "Sagar", "Sumangali"]
        }

        # Insert master data only if it doesn't exist
        for collection, items in master_data.items():
            col = await get_collection(collection)
            if col:
                for item in items:
                    try:
                        existing_item = await col.find_one({"name": item})
                        if not existing_item:
                            await col.insert_one({"name": item})
                    except Exception as e:
                        print(f"Error inserting {item} into {collection}: {e}")
                        # Continue to avoid crashing on duplicate or minor errors
        
        print("Database initialization completed successfully")
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Don't raise HTTPException here as it's not in a request context
        # Just log the error and continue
        print("Continuing without full database initialization...")

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
    if value:
        normalized_value = value.lower()
        existing_item = await collection.find_one({"name": {"$regex": f"^{normalized_value}$", "$options": "i"}})
        if not existing_item:
            raise ValueError(f"Invalid {field}: {value}. Must be one of the predefined {field}s.")
        return value  # Return the original value to preserve case if needed
    return value

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
        singer=singer
    )
    
    # Validate the fields
    try:
        if music_data.language:
            music_data.language = await validate_master_field("language", music_data.language, languages_col)
        if music_data.category:
            music_data.category = await validate_master_field("category", music_data.category, categories_col)
        if music_data.artist:
            music_data.artist = await validate_master_field("artist", music_data.artist, artists_col)
        if music_data.music_director:
            music_data.music_director = await validate_master_field("music_director", music_data.music_director, music_directors_col)
        if music_data.movie_name:
            music_data.movie_name = await validate_master_field("movie_name", music_data.movie_name, movies_col)
        if music_data.actor_name:
            music_data.actor_name = await validate_master_field("actor_name", music_data.actor_name, actors_col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return music_data

async def validate_music_update(music: MusicUpdate = Depends()):
    try:
        music.language = await validate_master_field("language", music.language, languages_col)
        music.category = await validate_master_field("category", music.category, categories_col)
        music.artist = await validate_master_field("artist", music.artist, artists_col)
        music.music_director = await validate_master_field("music_director", music.music_director, music_directors_col)
        music.movie_name = await validate_master_field("movie_name", music.movie_name, movies_col)
        music.actor_name = await validate_master_field("actor_name", music.actor_name, actors_col)
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
# Add this to your routes section

@app.post("/master/{collection}/add", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def add_to_master_list(collection: str, name: str = Form(...)):
    """Add a new item to a master collection (Admin only)"""
    
    # Validate collection exists
    col = await get_collection(collection)
    if col is None:
        available_collections = list((await get_collection_all()).keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid collection: '{collection}'. Available: {available_collections}"
        )
    
    # Validate name is not empty
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    try:
        # Check if already exists (case-insensitive)
        existing = await col.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
        if existing:
            return {
                "msg": f"{name} already exists in {collection}",
                "success": False,
                "existing_name": existing["name"]
            }
        
        # Insert new item
        result = await col.insert_one({"name": name})
        return {
            "msg": f"Successfully added '{name}' to {collection}",
            "success": True,
            "id": str(result.inserted_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to {collection}: {str(e)}")

# Bulk add endpoint
@app.post("/master/{collection}/bulk-add", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key), Depends(require_admin)])
async def bulk_add_to_master_list(collection: str, names: List[str]):
    """Add multiple items to a master collection (Admin only)"""
    
    col = await get_collection(collection)
    if col is None:
        available_collections = list((await get_collection_all()).keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid collection: '{collection}'. Available: {available_collections}"
        )
    
    results = {
        "added": [],
        "skipped": [],
        "errors": []
    }
    
    for name in names:
        name = name.strip()
        if not name:
            results["errors"].append({"name": name, "error": "Empty name"})
            continue
            
        try:
            # Check if already exists
            existing = await col.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
            if existing:
                results["skipped"].append({"name": name, "reason": "Already exists"})
                continue
            
            # Insert new item
            result = await col.insert_one({"name": name})
            results["added"].append({"name": name, "id": str(result.inserted_id)})
            
        except Exception as e:
            results["errors"].append({"name": name, "error": str(e)})
    
    return {
        "msg": f"Bulk add to {collection} completed",
        "success": True,
        "results": results,
        "summary": {
            "added_count": len(results["added"]),
            "skipped_count": len(results["skipped"]),
            "error_count": len(results["errors"])
        }
    }

@app.get("/master/{collection}", response_model=List[str], tags=["Master"], dependencies=[Depends(verify_api_key)])
async def get_master_list(collection: str):
    print(f"DEBUG - Requested collection: '{collection}'")
    all_collections = await get_collection_all()
    print(f"DEBUG - Available collections: {list(all_collections.keys())}")
    col = await get_collection(collection)
    if col is None:
        available_collections = list(all_collections.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid collection: '{collection}'. Available collections: {available_collections}"
        )
    try:
        items = await col.find().to_list(length=None)  # Remove length limit
        result = [item["name"] for item in items]
        print(f"DEBUG - Found {len(result)} items in {collection}: {result}")
        return result
    except Exception as e:
        print(f"DEBUG - Error fetching from {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data from {collection}: {str(e)}")

# Add a route to list all available collections
@app.get("/master", response_model=dict, tags=["Master"], dependencies=[Depends(verify_api_key)])
async def list_available_collections():
    """List all available master collections"""
    all_collections = await get_collection_all()
    collections_info = {}
    
    for name, col in all_collections.items():
        try:
            count = await col.count_documents({})
            collections_info[name] = {
                "count": count,
                "endpoint": f"/master/{name}"
            }
        except Exception as e:
            collections_info[name] = {
                "count": 0,
                "error": str(e),
                "endpoint": f"/master/{name}"
            }
    
    return {
        "available_collections": list(all_collections.keys()),
        "collections_info": collections_info
    }
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
async def upload_music(
    file: UploadFile = File(...),
    music_data: MusicCreate = Depends(get_music_data_from_form),
    current_user: dict = Depends(get_current_user)
):
    file_path, metadata = await save_uploaded_file(file, current_user)
    
    # Debug logging
    print(f"DEBUG - Received form data:")
    print(f"  language: '{music_data.language}' (type: {type(music_data.language)})")
    print(f"  genre: '{music_data.genre}' (type: {type(music_data.genre)})")
    print(f"  category: '{music_data.category}' (type: {type(music_data.category)})")
    
    print(f"DEBUG - Metadata from file:")
    print(f"  language: '{metadata.get('language')}' (type: {type(metadata.get('language'))})")
    print(f"  genre: '{metadata.get('genre')}' (type: {type(metadata.get('genre'))})")
    print(f"  category: '{metadata.get('category')}' (type: {type(metadata.get('category'))})")
    
    music_entry = {
        "name": music_data.name if music_data.name else (metadata.get("name") or f"Unnamed_{file.filename}"),
        "artist": music_data.artist if music_data.artist else (metadata.get("artist") or ""),
        "music_director": music_data.music_director if music_data.music_director else (metadata.get("music_director") or ""),
        "singer": music_data.singer if music_data.singer else (metadata.get("singer") or ""),
        "genre": music_data.genre if music_data.genre else (metadata.get("genre") or ""),
        "description": music_data.description or "",
        "language": music_data.language if music_data.language else (metadata.get("language") or ""),
        "category": music_data.category if music_data.category else (metadata.get("category") or ""),
        "is_keefy_original": music_data.is_keefy_original if music_data.is_keefy_original is not None else False,
        "movie_name": music_data.movie_name if music_data.movie_name else (metadata.get("movie_name") or ""),
        "actor_name": music_data.actor_name if music_data.actor_name else "",
        "uploaded_by": current_user["username"],
        "uploaded_at": datetime.utcnow(),
        "file_path": file_path,
        "is_completed": False
    }
    
    print(f"DEBUG - Final music_entry values:")
    print(f"  language: '{music_entry['language']}'")
    print(f"  genre: '{music_entry['genre']}'")
    print(f"  category: '{music_entry['category']}'")
    
    result = await music_col.insert_one(music_entry)
    stored_data = {k: str(v) if isinstance(v, ObjectId) else v for k, v in music_entry.items()}
    
    return {
        "msg": "Music uploaded successfully",
        "music_id": str(result.inserted_id),
        "file_path": file_path,
        "provided_data": music_data.dict(exclude_unset=True),
        "stored_data": stored_data,
        "debug_info": {
            "received_language": music_data.language,
            "received_genre": music_data.genre,
            "received_category": music_data.category,
            "final_language": music_entry['language'],
            "final_genre": music_entry['genre'],
            "final_category": music_entry['category']
        },
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
    try:
        obj_id = ObjectId(music_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid music ID format")
    
    update_data = music_data.dict(exclude_unset=True)
    update_data["is_completed"] = True
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

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    client.close()