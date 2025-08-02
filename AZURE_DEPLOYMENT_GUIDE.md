# Azure Deployment Guide

## Current Issue
Azure is not using the correct startup command and is still trying to run `python start_server.py` instead of `python azure_startup.py`.

## Files Created/Updated for Azure Deployment

### 1. `azure_startup.py` ✅
- Main entry point for Azure App Service
- Properly adds backend to Python path
- Handles port configuration from Azure environment

### 2. `startup.txt` ✅
- Contains: `python azure_startup.py`
- Should tell Azure which command to run

### 3. `web.config` ✅ (Updated)
- Configures HTTP Platform Handler
- Points to `azure_startup.py` as the startup script
- Sets proper environment variables

### 4. `.deployment` ✅ (New)
- Contains: `command = python azure_startup.py`
- Alternative way to specify startup command

### 5. `runtime.txt` ✅
- Specifies Python 3.11
- Should match your Azure App Service configuration

## Deployment Steps

### Step 1: Commit All Changes
```bash
git add .
git commit -m "Fix Azure deployment startup configuration"
git push origin main
```

### Step 2: Verify Azure App Service Configuration
1. Go to Azure Portal → Your App Service
2. Navigate to **Configuration** → **General settings**
3. Ensure **Stack** is set to **Python 3.11**
4. Ensure **Startup Command** is **empty** (let the files control it)

### Step 3: Trigger New Deployment
1. Go to **Deployment Center** in your Azure App Service
2. Click **Sync** or **Redeploy**
3. Wait for deployment to complete

### Step 4: Check Logs
1. Go to **Log stream** in Azure Portal
2. Look for these success messages:
   ```
   🚀 Starting LLM-Powered Document Analysis System on Azure
   📋 System Information:
      - Server will run on: http://0.0.0.0:8000
      - Hackathon endpoint: POST /hackrx/run
      - Health check: GET /health
   🔧 Starting server...
   ```

### Step 5: Test the Endpoint
Run the test script:
```bash
python test_azure_endpoint.py
```

## Troubleshooting

### If Still Getting "start_server.py" Error:
1. **Clear Azure Cache**: Go to Azure Portal → App Service → **Advanced Tools** → **Kudu** → **Console**
2. **Delete old files**: Remove any old `start_server.py` if it exists
3. **Restart App Service**: Go to **Overview** → **Restart**

### If Getting Permission Errors:
1. Check that all files have proper permissions
2. Ensure `azure_startup.py` is executable

### If Getting Import Errors:
1. Verify `backend/` directory structure is correct
2. Check that all dependencies are in `requirements.txt`

## Expected Success Logs
```
🚀 Starting LLM-Powered Document Analysis System on Azure
============================================================
📋 System Information:
   - Server will run on: http://0.0.0.0:8000
   - Hackathon endpoint: POST /hackrx/run
   - Health check: GET /health

🔧 Starting server...
============================================================
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://0.0.0.0:8000
```

## Test the Endpoint
After successful deployment, test with:
```bash
python test_azure_endpoint.py
```

This should return a 200 status code with answers to your questions. 