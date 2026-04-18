"""
Entry point: test.py
This file exists so the server can continue running `python3 test.py`.
It boots the modular Voice Engine from app/main.py using uvicorn's
lazy string-based import to avoid circular import issues.
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8085, reload=False)
