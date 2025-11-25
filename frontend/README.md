# PDF to Video Generator - Frontend

React frontend for the PDF to Video Generator application.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The app will run on http://localhost:3000

**Note:** If you see a deprecation warning about `util._extend`, it's harmless and comes from a Node.js dependency. It doesn't affect functionality and can be safely ignored. This will be fixed in future dependency updates.

## Environment Variables

Create a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:8000
```

## Features

- Upload PDF files with drag-and-drop support
- Configure page range for video generation
- Real-time job status tracking
- Generate book summaries
- Generate summary videos
- Download generated videos and summaries

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Deployment to Vercel

### Prerequisites
1. A Vercel account (sign up at [vercel.com](https://vercel.com))
2. Vercel CLI installed (optional, for CLI deployment):
   ```bash
   npm i -g vercel
   ```

### Deployment Steps

#### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to GitHub/GitLab/Bitbucket** (if not already done)

2. **Go to [vercel.com](https://vercel.com)** and sign in

3. **Click "Add New Project"**

4. **Import your repository** containing the frontend code

5. **Configure the project:**
   - **Framework Preset:** Create React App (or leave as "Other")
   - **Root Directory:** `frontend` (if your repo root is the parent directory)
   - **Build Command:** `npm run build`
   - **Output Directory:** `build`
   - **Install Command:** `npm install`

6. **Add Environment Variable:**
   - Go to **Settings** → **Environment Variables**
   - Add a new variable:
     - **Name:** `REACT_APP_API_URL`
     - **Value:** `http://54.242.255.232:8000`
     - **Environment:** Production, Preview, Development (select all)

7. **Click "Deploy"**

#### Option 2: Deploy via Vercel CLI

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   vercel
   ```

4. **Set environment variable:**
   ```bash
   vercel env add REACT_APP_API_URL
   # When prompted, enter: http://54.242.255.232:8000
   ```

5. **Redeploy with the new environment variable:**
   ```bash
   vercel --prod
   ```

### Important Notes

- **CORS Configuration:** Make sure your backend at `http://54.242.255.232:8000` has CORS enabled to allow requests from your Vercel domain. The backend should allow the origin of your Vercel deployment.

- **HTTPS/HTTP:** If your backend uses HTTP (not HTTPS), you may encounter CORS issues in production. Consider:
  - Setting up HTTPS for your backend (recommended)
  - Or configuring CORS on your backend to allow your Vercel domain

- **Environment Variables:** The `REACT_APP_API_URL` environment variable is baked into the build at build time, so you'll need to redeploy if you change it.

### After Deployment

Once deployed, Vercel will provide you with a URL like `https://your-app.vercel.app`. Your frontend will now communicate with your AWS backend at `http://54.242.255.232:8000`.

