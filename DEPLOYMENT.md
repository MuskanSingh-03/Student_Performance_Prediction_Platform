# Cloud Deployment Guide — Student Performance Prediction Platform

This guide explains how to deploy the platform to cloud hosting providers (**Render**, **Railway**, **Vercel**, **PythonAnywhere**, or **Heroku**).

---

## 🛠️ Issues Fixed for Cloud Deployment

If your deployment was previously failing or showing a "502 Bad Gateway" / "Connection Refused" / "Build Failed", the following key fixes have been applied:

1. **Host Binding (`0.0.0.0`)**: 
   - Previously bound to `127.0.0.1` (which only allows connections from within the container). Now binds to `0.0.0.0` so external traffic can reach the web app.
2. **Dynamic Port Binding (`$PORT`)**: 
   - Cloud providers (Render, Heroku, Railway) assign a dynamic port via the `PORT` environment variable. The app now reads `int(os.environ.get("PORT", 5000))`.
3. **Production WSGI Server (`gunicorn` & `waitress`)**:
   - Added `gunicorn` and `waitress` to `requirements.txt`.
   - Added standard `Procfile` (`web: gunicorn app:app`).
   - Added `wsgi.py` entry point.
4. **Self-Healing Startup**:
   - If trained models (`.joblib`) or dataset (`.csv`) are not committed to Git, the app automatically generates the dataset and trains the models on first startup instead of crashing with a `FileNotFoundError`.
5. **Persistent Database Initialization**:
   - SQLite database schema and initial demo users (`amit` and `muskan`) are auto-created if not already present.

---

## 🚀 Option 1: Deploy to Render (Recommended — Free & Easiest)

Render is the best platform for Flask + Machine Learning apps.

### Steps:
1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for deployment"
   git remote add origin https://github.com/YOUR_USERNAME/student-performance-prediction.git
   git push -u origin main
   ```

2. **Go to [Render.com](https://render.com/) and Sign Up / Log In.**

3. **Click "New +" &rarr; "Web Service".**

4. **Connect your GitHub repository.**

5. **Configure Settings:**
   - **Name:** `student-performance-predictor` (or your choice)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`

6. **Click "Create Web Service".**
   - Render will build the environment and start the application.
   - You will receive a live URL: `https://your-app-name.onrender.com`

---

## 🚄 Option 2: Deploy to Railway

1. Push your code to GitHub.
2. Go to [Railway.app](https://railway.app/).
3. Click **"New Project" &rarr; "Deploy from GitHub repo"**.
4. Select your repository.
5. Railway will automatically detect the `Procfile` and `requirements.txt` and deploy with Gunicorn.
6. Under **Settings &rarr; Networking**, click **"Generate Domain"** to get your public URL.

---

## ▲ Option 3: Deploy to Vercel

1. Install the Vercel CLI or connect via GitHub on [Vercel.com](https://vercel.com/).
2. The included `vercel.json` already routes requests to `app.py`.
3. Run:
   ```bash
   vercel
   ```
4. Follow the CLI prompts to deploy.

---

## 🐍 Option 4: Deploy to PythonAnywhere

1. Create a free account on [PythonAnywhere.com](https://www.pythonanywhere.com/).
2. Open a Bash console and clone your repo or upload files.
3. Create a virtual environment:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 myenv
   pip install -r requirements.txt
   ```
4. In the **Web** tab:
   - Set Python version to 3.10.
   - Set Virtualenv path to `/home/yourusername/.virtualenvs/myenv`.
   - In the **WSGI configuration file**, edit it to:
     ```python
     import sys
     path = '/home/yourusername/Student Performance Prediction Platform'
     if path not in sys.path:
         sys.path.append(path)
     from app import app as application
     ```
5. Click **Reload your web app**.

---

## 💻 Option 5: Run Production Server Locally or on Private Network

To serve the app in production on your local network or server:

### On Windows (using Waitress):
```powershell
python -c "from waitress import serve; from app import app; print('Serving on http://0.0.0.0:5000'); serve(app, host='0.0.0.0', port=5000)"
```

### On Linux / macOS (using Gunicorn):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

---

## 🔑 Default Credentials on Deployed App

| Role | Username / Email | Password | Full Name Displayed |
|:---|:---:|:---:|:---|
| **Faculty Mentor** | `amit` / `amit@niet.co.in` | `password123` | **Amit Yadav** |
| **Student** | `muskan` / `muskan@niet.co.in` | `password123` | **Muskan Gola** |
| **Administrator** | `admin` / `admin@niet.co.in` | `admin123` | **Dean Academics** |

*(You can also use the **`/register`** page on your deployed site to register any new faculty member or student!)*
