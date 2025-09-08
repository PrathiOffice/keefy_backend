import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def test_mongodb_connection():
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("DB_NAME")]
        
        # Test connection
        await client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Test reading from users collection
        users_count = await db.users.count_documents({})
        print(f"👥 Users in database: {users_count}")
        
        # Test reading from music collection
        music_count = await db.music.count_documents({})
        print(f"🎵 Songs in database: {music_count}")
        
        # Get the admin user
        admin_user = await db.users.find_one({"username": "admin"})
        if admin_user:
            print(f"👤 Found admin user: {admin_user['email']}")
        
        # Get sample music
        sample_music = await db.music.find_one()
        if sample_music:
            print(f"🎶 Found sample song: {sample_music['name']} by {sample_music['artist']}")
        
        client.close()
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_mongodb_connection())