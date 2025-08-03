from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    """Authenticate a user with username and password."""
    if request.username == "admin" and request.password == "password":  # Demo credentials
        return {"message": "Login successful", "token": "dummy-jwt-token"}
    raise HTTPException(status_code=401, detail="Invalid credentials")