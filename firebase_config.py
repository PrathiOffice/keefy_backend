import firebase_admin
from firebase_admin import credentials, firestore, storage

# Load credentials
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
