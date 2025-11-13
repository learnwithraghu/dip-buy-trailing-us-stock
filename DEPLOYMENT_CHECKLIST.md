# Pre-Deployment Checklist

Before deploying to Streamlit Cloud, verify the following:

## ✅ Required Files

- [x] `app.py` - Main application file
- [x] `requirements.txt` - All Python dependencies listed
- [x] `.gitignore` - Excludes CSV files and virtual environments
- [x] `.streamlit/config.toml` - Streamlit configuration (optional but recommended)
- [x] `README.md` - Documentation

## ✅ Code Verification

- [x] All imports are available in `requirements.txt`
- [x] CSV file path is relative (works in cloud)
- [x] No hardcoded absolute paths
- [x] Error handling for file operations
- [x] Logging configured properly

## ✅ Git Setup

Before pushing to GitHub:

```bash
# Check what will be committed
git status

# Make sure these are NOT committed:
# - stock_tracker.csv
# - venv/
# - .venv/
# - __pycache__/

# Make sure these ARE committed:
# - app.py
# - requirements.txt
# - README.md
# - .gitignore
# - .streamlit/config.toml
```

## ✅ Test Locally First

1. Test the app runs without errors:
   ```bash
   streamlit run app.py
   ```

2. Test adding stocks to tracker
3. Test removing stocks from tracker
4. Verify CSV file is created/updated
5. Restart app and verify tracker data loads

## ✅ Ready to Deploy

Once all checks pass:
1. Push to GitHub
2. Deploy on Streamlit Cloud
3. Test the deployed app
4. Verify tracker persistence works

## Common Issues

**Issue**: App won't deploy
- **Solution**: Check `requirements.txt` has all dependencies

**Issue**: CSV file not persisting
- **Solution**: CSV is created automatically in cloud, verify file permissions

**Issue**: Import errors
- **Solution**: Ensure all packages in `requirements.txt` match imports in `app.py`

