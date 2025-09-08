from sqlalchemy import Integer, String, ForeignKey, Table, Text, DateTime
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime
from .database import Base

playlist_tracks = Table(
    "playlist_tracks", Base.metadata,
    mapped_column("playlist_id", ForeignKey("playlists.id"), primary_key=True),
    mapped_column("track_id", ForeignKey("tracks.id"), primary_key=True),
)

user_likes = Table(
    "user_likes", Base.metadata,
    mapped_column("user_id", ForeignKey("users.id"), primary_key=True),
    mapped_column("track_id", ForeignKey("tracks.id"), primary_key=True),
)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    playlists = relationship("Playlist", back_populates="owner")
    likes = relationship("Track", secondary=user_likes, back_populates="liked_by")

class Artist(Base):
    __tablename__ = "artists"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    bio: Mapped[str | None] = mapped_column(Text)
    albums = relationship("Album", back_populates="artist")

class Album(Base):
    __tablename__ = "albums"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    artist_id: Mapped[int] = mapped_column(ForeignKey("artists.id"))
    artist = relationship("Artist", back_populates="albums")
    tracks = relationship("Track", back_populates="album")

class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    album_id: Mapped[int | None] = mapped_column(ForeignKey("albums.id"))
    artist_id: Mapped[int | None] = mapped_column(ForeignKey("artists.id"))
    duration_sec: Mapped[int | None] = mapped_column(Integer)
    filename: Mapped[str | None] = mapped_column(String(400))  # stored under MEDIA_DIR

    album = relationship("Album", back_populates="tracks")
    artist = relationship("Artist")
    liked_by = relationship("User", secondary=user_likes, back_populates="likes")

class Playlist(Base):
    __tablename__ = "playlists"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    owner = relationship("User", back_populates="playlists")
    tracks = relationship("Track", secondary=playlist_tracks, lazy="select")