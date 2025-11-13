# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Prepare Your Repository

Make sure your repository has:
- ✅ `app.py` (main application file)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `.gitignore` (excludes `stock_tracker.csv` and other unnecessary files)
- ✅ `.streamlit/config.toml` (optional, for app configuration)

### 2. Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "DJIA Stock Scanner App - Ready for deployment"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to GitHub
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a unique name (e.g., `djia-stock-scanner`)
5. Click **"Deploy!"**

### 4. Wait for Deployment

- Streamlit Cloud will install dependencies from `requirements.txt`
- The app will be available at: `https://your-app-name.streamlit.app`
- First deployment may take 2-3 minutes

## Important Notes

### Data Persistence

- The `stock_tracker.csv` file is created automatically in the cloud environment
- Data persists as long as the app is deployed
- Each user session has its own tracker data (stored in the app's working directory)
- If you redeploy or the app restarts, the CSV file will be recreated if it doesn't exist

### Environment Variables (Optional)

If you need to customize settings, you can add environment variables in Streamlit Cloud:
- Go to your app settings
- Add environment variables if needed
- Currently, all settings are in the code (capital, allocation, etc.)

### Updating Your App

1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push
   ```
3. Streamlit Cloud will automatically redeploy your app

### Troubleshooting

**App won't deploy?**
- Check that `requirements.txt` has all dependencies
- Verify `app.py` is in the root directory
- Check the deployment logs in Streamlit Cloud dashboard

**CSV file not persisting?**
- The CSV file is stored in the app's working directory
- It persists across app restarts but may be reset if you redeploy
- Consider using a database (like SQLite or external DB) for production use

**Dependencies not installing?**
- Check `requirements.txt` format
- Ensure all package versions are compatible
- Check deployment logs for specific errors

## File Structure for Deployment

```
your-repo/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── .gitignore           # Git ignore rules
├── .streamlit/          # Streamlit config (optional)
│   └── config.toml
└── stock_tracker.csv    # Created automatically (not in git)
```

## Next Steps

After deployment:
1. Test the app functionality
2. Add stocks to tracker and verify persistence
3. Share the app URL with others
4. Monitor usage and performance

For more help, visit [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)

