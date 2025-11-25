# Vercel Deployment Guide

This guide will help you deploy the frontend of the PDF to Video Generator application to Vercel.

## Prerequisites

1. **GitHub/GitLab/Bitbucket Account**: Your code should be in a Git repository
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com) (free tier is sufficient)
3. **Backend URL**: Your backend is already deployed at `http://54.242.255.232:8000`

## Quick Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended for First Time)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Go to [vercel.com](https://vercel.com)** and sign in

3. **Click "Add New Project"** or **"Import Project"**

4. **Import your Git repository**:
   - Select your repository from the list
   - Click "Import"

5. **Configure Project Settings**:
   - **Framework Preset**: Leave as "Other" or select "Create React App"
   - **Root Directory**: If your repo root contains both frontend and backend, set this to `frontend`
   - **Build Command**: `npm run build` (should auto-detect)
   - **Output Directory**: `build` (should auto-detect)
   - **Install Command**: `npm install` (should auto-detect)

6. **Add Environment Variable**:
   - Before clicking "Deploy", go to **Environment Variables** section
   - Click **"Add"** and add:
     - **Name**: `REACT_APP_API_URL`
     - **Value**: `http://54.242.255.232:8000`
     - **Environment**: Select all (Production, Preview, Development)

7. **Click "Deploy"**

8. **Wait for deployment** (usually 1-3 minutes)

9. **Your app is live!** Vercel will provide you with a URL like `https://your-app-name.vercel.app`

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

3. **Login to Vercel**:
   ```bash
   vercel login
   ```
   - This will open a browser window for authentication

4. **Deploy to Vercel**:
   ```bash
   vercel
   ```
   - Follow the prompts:
     - Set up and deploy? **Yes**
     - Which scope? (Select your account)
     - Link to existing project? **No** (for first deployment)
     - Project name? (Press Enter for default or enter a custom name)
     - Directory? (Press Enter, it should detect `frontend`)
     - Override settings? **No** (for first time)

5. **Set Environment Variable**:
   ```bash
   vercel env add REACT_APP_API_URL
   ```
   - When prompted, enter: `http://54.242.255.232:8000`
   - Select environments: Production, Preview, Development (select all)

6. **Redeploy with environment variable**:
   ```bash
   vercel --prod
   ```

## Post-Deployment Configuration

### Custom Domain (Optional)

1. Go to your project in Vercel dashboard
2. Click **Settings** → **Domains**
3. Add your custom domain
4. Follow DNS configuration instructions

### Environment Variables Management

To update the API URL later:
1. Go to **Settings** → **Environment Variables**
2. Edit `REACT_APP_API_URL`
3. Redeploy the project

## Important Notes

### CORS Configuration

Your backend at `http://54.242.255.232:8000` already has CORS configured to allow all origins (`allow_origins=["*"]`), so your Vercel deployment should work without additional backend changes.

However, for better security in production, consider updating your backend CORS to only allow your Vercel domain:

```python
# In app/api/main.py, update line 40:
allow_origins=[
    "https://your-app.vercel.app",
    "https://your-app.vercel.app/*",
    # Add your custom domain if you have one
],
```

### HTTPS/HTTP Mixed Content

Since your backend uses HTTP (`http://54.242.255.232:8000`) and Vercel serves over HTTPS, you may encounter CORS or mixed content issues. If you experience problems:

1. **Option A**: Set up HTTPS for your backend (recommended)
   - Use a reverse proxy like Nginx with Let's Encrypt
   - Or use AWS Application Load Balancer with SSL certificate

2. **Option B**: Configure backend to allow your Vercel domain specifically
   - Update CORS settings as shown above

### Build Optimization

Vercel automatically:
- Detects React apps
- Runs `npm install` and `npm run build`
- Serves the `build` folder
- Handles routing for single-page applications

## Troubleshooting

### Build Fails

- Check build logs in Vercel dashboard
- Ensure all dependencies are in `package.json`
- Verify Node.js version (Vercel uses Node 18.x by default)

### API Calls Fail

- Verify `REACT_APP_API_URL` is set correctly
- Check browser console for CORS errors
- Ensure backend is accessible from the internet
- Check backend logs for incoming requests

### Environment Variables Not Working

- Remember: React environment variables must start with `REACT_APP_`
- After changing environment variables, you must redeploy
- Environment variables are baked into the build at build time

### Routing Issues

- The `vercel.json` file handles SPA routing
- All routes should redirect to `index.html`

## Updating Your Deployment

After making changes to your frontend:

1. **Push to Git**:
   ```bash
   git add .
   git commit -m "Update frontend"
   git push
   ```

2. **Vercel will automatically deploy** (if connected to Git)
   - Or manually trigger from Vercel dashboard
   - Or run `vercel --prod` from CLI

## Monitoring

- **Deployment Logs**: Available in Vercel dashboard
- **Analytics**: Available in Vercel dashboard (may require upgrade)
- **Error Tracking**: Consider adding Sentry or similar service

## Support

- Vercel Documentation: https://vercel.com/docs
- Vercel Community: https://github.com/vercel/vercel/discussions

