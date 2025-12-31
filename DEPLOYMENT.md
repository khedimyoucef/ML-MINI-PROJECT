# Deployment Guide for Streamlit Cloud

This guide explains how to deploy your ML Mini-Project to Streamlit Cloud with automatic downloading from Hugging Face.

## Prerequisites

1. **Hugging Face Repository**: Your models and datasets uploaded to `khedim/ML-MINI-PROJECT`
2. **Hugging Face Token**: Get from https://huggingface.co/settings/tokens
3. **GitHub Repository**: Your code pushed to GitHub (without large files)

## Step 1: Prepare Your Repository

Make sure your `.gitignore` excludes large files:

```
DS2GROCERIES/
DS3RECIPES/
models/
features/
src/token.txt
.streamlit/secrets.toml
```

## Step 2: Push to GitHub

```bash
git add .
git commit -m "Add Hugging Face auto-download support"
git push origin main
```

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Set main file path: `app.py`
5. Click "Advanced settings"

## Step 4: Configure Secrets

In the "Secrets" section, add:

```toml
HF_TOKEN = "your_actual_huggingface_token_here"
```

Replace `your_actual_huggingface_token_here` with your real token from Hugging Face.

## Step 5: Deploy

Click "Deploy!" and wait for the app to start.

## How It Works

1. When the app starts, it calls `ensure_models_available()` from `src/hf_utils.py`
2. This function checks if required directories exist and have files
3. If anything is missing, it downloads from your HF repo using `snapshot_download()`
4. The download only happens once - subsequent runs use cached files

## Troubleshooting

### "Error downloading from Hugging Face"

- **Check your HF_TOKEN**: Make sure it's correctly set in Streamlit secrets
- **Check repo visibility**: If your HF repo is private, the token must have read access
- **Check repo name**: Ensure it matches `khedim/ML-MINI-PROJECT`

### "Repository not found"

- Make sure your HF repository is public, OR
- Provide a valid HF_TOKEN with access to the private repo

### Files not downloading

- Check the Streamlit logs for specific error messages
- Verify your HF repo contains the required directories: `models/`, `features/`, `DS2GROCERIES/`, `DS3RECIPES/`

## Local Testing

To test the download functionality locally:

```bash
# Create a token file (for local dev only)
echo "your_token_here" > src/token.txt

# Run the app
streamlit run app.py
```

The app will automatically download missing files on first run.

## Security Notes

- **Never commit your token** to Git
- The `.gitignore` is configured to exclude `src/token.txt` and `.streamlit/secrets.toml`
- For deployment, always use Streamlit secrets, not token files
