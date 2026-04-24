@echo off
cd /d "%~dp0"
call conda activate Rag_env
uvicorn Api:app --reload --host 127.0.0.1 --port 8000
