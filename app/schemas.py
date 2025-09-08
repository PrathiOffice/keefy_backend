from pydantic import BaseModel, EmailStr
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    display_name: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    display_name: str
    class Config:
        from_attributes = True

class ArtistIn(BaseModel):
    name: str
    bio: Optional[str] = None

class ArtistOut(BaseModel):
    id: int
    name: str
    bio: Optional[str] = None
    class Config:
        from_attributes = True

class AlbumIn(BaseModel):
    title: str
    artist_id: int

class AlbumOut(BaseModel):
    id: int
    title: str
    artist_id: int
    class Config:
        from_attributes = True

class TrackIn(BaseModel):
    title: str
    album_id: Optional[int] = None
    artist_id: Optional[int] = None
    duration_sec: Optional[int] = None

class TrackOut(BaseModel):
    id: int
    title: str
    album_id: Optional[int] = None
    artist_id: Optional[int] = None
    duration_sec: Optional[int] = None
    filename: Optional[str] = None
    class Config:
        from_attributes = True

class PlaylistIn(BaseModel):
    name: str

class PlaylistOut(BaseModel):
    id: int
    name: str
    owner_id: int
    class Config:
        from_attributes = True