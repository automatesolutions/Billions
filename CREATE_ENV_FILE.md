# Create .env.local File

## Quick Fix for Frontend Errors

I fixed the CSS error! Now create this file to fix the auth error:

### Step 1: Create `web/.env.local` file

**Path**: `C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Billions\web\.env.local`

**Contents**:
```env
# BILLIONS Web App - Local Development

NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=billions-dev-secret-12345

NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Step 2: Restart Frontend

Stop the frontend (Ctrl+C) and restart:
```powershell
pnpm dev
```

---

## ✅ That's it!

The frontend will now work without errors!

(Google OAuth can be set up later)

