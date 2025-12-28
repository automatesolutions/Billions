@echo off
echo Starting BILLIONS Frontend...
echo.

cd /d "%~dp0web"
echo Changed to web directory
echo.

echo Starting Next.js development server
echo Frontend will be at http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

pnpm dev
