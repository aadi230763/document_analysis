# Azure Startup Command Guide

## 🎯 **Current Issue**
The Azure container is failing to start with exit code 1. Based on the logs and codebase analysis, here are the correct startup commands to use.

## 📋 **Recommended Startup Commands**

### **Option 1: Use the Shell Script (Recommended)**
```
bash startup.sh
```

### **Option 2: Direct Python Command**
```
python startup_direct.py
```

### **Option 3: Direct Backend App**
```
cd /home/site/wwwroot/backend && python app.py
```

### **Option 4: Gunicorn (Production)**
```
cd /home/site/wwwroot/backend && gunicorn --bind 0.0.0.0:8000 app:app
```

## 🔧 **How to Set the Startup Command in Azure**

### **Method 1: Azure Portal**
1. Go to your Azure App Service
2. Navigate to **Configuration** → **General settings**
3. Set **Startup Command** to one of the options above
4. Click **Save**

### **Method 2: Azure CLI**
```bash
az webapp config set --name bajaj --resource-group your-resource-group --startup-file "bash startup.sh"
```

### **Method 3: GitHub Actions (Recommended)**
Add this to your `.github/workflows/deploy.yml`:
```yaml
- name: Set startup command
  run: |
    az webapp config set --name bajaj --resource-group your-resource-group --startup-file "bash startup.sh"
```

## 📁 **File Structure Analysis**

Based on the codebase, here's the correct file structure:

```
/home/site/wwwroot/
├── startup.sh                    # Shell script startup
├── startup_direct.py            # Python startup script
├── start_server.py              # Alternative startup
├── backend/
│   ├── app.py                   # Main Flask application
│   ├── config.py                # Configuration
│   ├── utils/                   # Utility modules
│   │   ├── medical_terms.py
│   │   ├── query_expander.py
│   │   ├── rule_engine.py
│   │   └── pinecone_manager.py
│   └── requirements.txt
└── requirements.txt
```

## 🚀 **Expected Behavior**

### **Successful Startup Logs:**
```
🚀 Starting LLM-Powered Document Analysis System on Azure
============================================================
📋 System Information:
   - Server will run on: http://0.0.0.0:8000
   - Hackathon endpoint: POST /hackrx/run
   - Health check: GET /health
   - Python path: /home/site/wwwroot/backend

🔧 Starting server...
============================================================
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://0.0.0.0:8000
```

### **Available Endpoints:**
- `GET /health` - Health check
- `POST /hackrx/run` - Main hackathon endpoint
- `POST /query` - General query endpoint
- `POST /batch_query` - Batch processing

## 🔍 **Troubleshooting**

### **If startup still fails:**

1. **Check the logs** for specific error messages
2. **Verify file permissions** - ensure startup scripts are executable
3. **Test locally** - run the startup command locally to verify it works
4. **Check dependencies** - ensure all required packages are installed

### **Common Issues:**

1. **Import Errors**: The backend app imports from `utils.*` modules
2. **Path Issues**: Make sure the backend directory is in the Python path
3. **Port Conflicts**: Ensure the app binds to `0.0.0.0:8000`
4. **Environment Variables**: Check if required env vars are set

## ✅ **Recommended Action**

**Set your Azure startup command to:**
```
bash startup.sh
```

This will:
1. Set the correct Python path
2. Change to the backend directory
3. Set production environment variables
4. Start the Flask app with proper error handling

## 📞 **If Issues Persist**

1. Check the Azure Log Stream for specific error messages
2. Verify that all files are deployed correctly
3. Test the startup command locally
4. Check if all dependencies are installed

---

**Last Updated**: August 2, 2025
**Status**: ✅ **READY FOR DEPLOYMENT**
**Recommended Command**: `bash startup.sh` 