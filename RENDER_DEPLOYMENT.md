# Render.com Deployment Guide

## Prerequisites
- A Render.com account (sign up at https://render.com)
- Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)

## Deployment Steps

### 1. Prepare Your Repository
- Ensure all configuration files are committed:
  - `requirements.txt` - Python dependencies
  - `render.yaml` - Render build configuration
  - `Procfile` - Alternative start command
  - `runtime.txt` - Python version specification

### 2. Create a New Web Service on Render

1. Go to https://dashboard.render.com
2. Click "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: `aerial-image-segmentation` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Choose appropriate tier (Free tier available for testing)

### 3. Set Environment Variables

In Render dashboard, add the following environment variables:

| Key | Value | Notes |
|-----|-------|-------|
| `SECRET_KEY` | Auto-generated | Render can generate this automatically for security |
| `PORT` | (auto) | Render automatically sets this |
| `DATABASE_URL` | PostgreSQL URL | See database setup below |

**To generate SECRET_KEY automatically:**
1. In the "Environment" section, Render will suggest auto-generating it
2. Click the button to generate a secure key

### 4. Set Up Database (Recommended for Production)

For production, switch from SQLite to PostgreSQL:

1. In Render dashboard, create a new PostgreSQL database:
   - Click "New +" → "PostgreSQL"
   - Name: `aerial-db`
   - Region: Same as your web service
   - PostgreSQL Version: 15

2. Once created, Render will automatically populate `DATABASE_URL` environment variable

3. The app will automatically use PostgreSQL when `DATABASE_URL` is set

### 5. Deploy

1. Push your changes to your Git repository
2. Render will automatically detect the push and start deployment
3. Monitor the deployment logs in the Render dashboard
4. Once deployed, your app will be available at: `https://aerial-image-segmentation.onrender.com`

## Important Notes

### Model File (resnetunet_aerial.pth)
- The model file (150+ MB) is already in `.gitignore`
- **Option 1 (Not Recommended)**: Add to Git LFS if < 1GB
- **Option 2 (Recommended)**: Download during build time
  - Modify build command or create a script to download from a storage service
  - Use AWS S3, Google Cloud Storage, or similar

### Upload Storage
- Currently uses local `static/uploads/` directory
- On Render, files are ephemeral (lost when service restarts)
- **For production**:
  1. Use external storage (AWS S3, Cloudinary, etc.)
  2. Or use Render's persistent disk (modify configuration)

### Database Persistence
- SQLite won't persist across Render restarts
- Use PostgreSQL for persistent data (included in configuration)

## Local Testing Before Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file (optional)
cp .env.example .env

# Run locally
python app.py

# Or with gunicorn
gunicorn app:app
```

## Troubleshooting

### Build Fails
- Check `requirements.txt` syntax
- Ensure all dependencies are listed
- Check build logs in Render dashboard

### App Crashes After Deploy
- Check runtime logs in Render dashboard
- Verify environment variables are set correctly
- Ensure `SECRET_KEY` is set (not empty)

### Model Not Found
- Ensure `resnetunet_aerial.pth` is accessible
- Consider using storage service for model distribution

### File Uploads Lost
- This is expected behavior with ephemeral storage
- Set up persistent disk or external storage solution

## Environment Variables Reference

```
SECRET_KEY          - Flask session secret (required, auto-generated on Render)
PORT                - Server port (automatically set by Render, default 5000)
DATABASE_URL        - Database connection string (SQLite by default, PostgreSQL recommended for production)
```

## Additional Resources
- Render Documentation: https://render.com/docs
- Flask Deployment: https://flask.palletsprojects.com/deployment/
- Gunicorn: https://gunicorn.org/
