# Google OAuth Setup Guide

## Prerequisites
- Google Account
- Access to Google Cloud Console

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "New Project"
4. Enter project name: "BILLIONS" (or your preferred name)
5. Click "Create"

## Step 2: Enable Google+ API

1. In the Google Cloud Console, select your project
2. Go to "APIs & Services" → "Library"
3. Search for "Google+ API"
4. Click on it and click "Enable"

## Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" → "OAuth consent screen"
2. Select "External" user type
3. Click "Create"
4. Fill in the required information:
   - **App name**: BILLIONS
   - **User support email**: Your email
   - **Developer contact information**: Your email
5. Click "Save and Continue"
6. **Scopes**: Click "Add or Remove Scopes"
   - Add: `email`
   - Add: `profile`
   - Add: `openid`
7. Click "Save and Continue"
8. **Test users** (for development):
   - Add your email address
   - Add any other test users
9. Click "Save and Continue"
10. Review and click "Back to Dashboard"

## Step 4: Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Select "Web application"
4. Enter a name: "BILLIONS Web App"
5. **Authorized JavaScript origins**:
   - Add: `http://localhost:3000`
   - For production, add: `https://yourdomain.com`
6. **Authorized redirect URIs**:
   - Add: `http://localhost:3000/api/auth/callback/google`
   - For production, add: `https://yourdomain.com/api/auth/callback/google`
7. Click "Create"
8. **Important**: Copy the Client ID and Client Secret

## Step 5: Configure Environment Variables

### Development (.env.local)

Create or update `web/.env.local`:

```bash
# NextAuth
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-generated-secret-here

# Google OAuth
GOOGLE_CLIENT_ID=your_client_id_here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret_here
```

### Generate NEXTAUTH_SECRET

Run this command to generate a secure secret:

```bash
# On macOS/Linux
openssl rand -base64 32

# On Windows (PowerShell)
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

## Step 6: Test Authentication

1. Start the backend API:
   ```bash
   python -m uvicorn api.main:app --reload
   ```

2. Start the frontend:
   ```bash
   cd web
   pnpm dev
   ```

3. Navigate to `http://localhost:3000/login`

4. Click "Sign in with Google"

5. You should be redirected to Google's OAuth consent screen

6. After granting permission, you should be redirected back to `/dashboard`

## Troubleshooting

### Error: "redirect_uri_mismatch"
- Check that the redirect URI in Google Cloud Console matches exactly:
  - `http://localhost:3000/api/auth/callback/google` (dev)
  - No trailing slashes
  - Correct protocol (http vs https)

### Error: "access_denied"
- Make sure your email is added as a test user
- Check OAuth consent screen configuration
- Verify app is not in review/suspended state

### Error: "invalid_client"
- Verify GOOGLE_CLIENT_ID is correct
- Verify GOOGLE_CLIENT_SECRET is correct
- Check for extra spaces or newlines in .env.local

### Session Issues
- Clear browser cookies
- Check NEXTAUTH_SECRET is set
- Verify NEXTAUTH_URL matches your development URL

## Production Deployment

### Additional Steps for Production:

1. **Verify OAuth Consent Screen**:
   - Go through app verification process
   - Add privacy policy URL
   - Add terms of service URL

2. **Update Authorized URLs**:
   - Add production domain to authorized origins
   - Add production callback URL

3. **Environment Variables**:
   ```bash
   NEXTAUTH_URL=https://yourdomain.com
   NEXTAUTH_SECRET=different-secret-for-production
   GOOGLE_CLIENT_ID=same-as-dev-or-create-new
   GOOGLE_CLIENT_SECRET=same-as-dev-or-create-new
   ```

4. **Security Best Practices**:
   - Never commit `.env.local` to git
   - Use different secrets for dev/staging/production
   - Rotate secrets periodically
   - Use environment-specific OAuth clients if possible

## References

- [NextAuth.js Documentation](https://next-auth.js.org/)
- [Google OAuth 2.0 Setup](https://support.google.com/cloud/answer/6158849)
- [NextAuth Google Provider](https://next-auth.js.org/providers/google)

## Support

If you encounter issues:
1. Check the console for error messages
2. Verify all environment variables
3. Review Google Cloud Console settings
4. Check NextAuth.js debug logs (set DEBUG=true)

---

**Last Updated**: 2025-10-10
**Status**: Ready for Development

