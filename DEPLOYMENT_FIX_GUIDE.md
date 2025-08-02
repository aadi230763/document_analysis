# Azure Deployment Fix Guide

## 🚨 **Issue Identified: Pip Install Failure**

The GitHub Actions deployment was failing because of problematic dependencies in `requirements.txt`:
- **PyTorch installation** with `-f` flag was causing issues
- **System dependencies** like `pytesseract` and `pdf2image` require additional setup

## ✅ **Fixes Applied**

### 1. **Fixed requirements.txt**
- Removed problematic PyTorch installation
- Removed system-dependent packages (`pytesseract`, `pdf2image`)
- Added specific version numbers for better compatibility
- Commented out optional dependencies (`redis`, `celery`)

### 2. **Fixed backend/requirements.txt**
- Applied the same fixes to backend requirements
- Ensured consistency between root and backend requirements

## 🚀 **Next Steps**

### **Step 1: Commit and Push Changes**
```bash
git add .
git commit -m "Fix deployment issues: remove problematic dependencies"
git push origin main
```

### **Step 2: Monitor GitHub Actions**
1. Go to your GitHub repository
2. Check the **Actions** tab
3. Monitor the deployment workflow
4. Look for successful deployment

### **Step 3: After Successful Deployment**
Once deployment succeeds, the startup command issue will be resolved because:
- The new files (`startup.sh`, `startup.py`, etc.) will be deployed
- Azure will use the correct startup command

## 📋 **Expected Success Logs**

After successful deployment, you should see in Azure Log Stream:
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

## 🔧 **If Deployment Still Fails**

If the deployment still fails, try these additional steps:

### **Option 1: Use Azure-optimized requirements**
Rename `requirements-azure.txt` to `requirements.txt` for a minimal deployment.

### **Option 2: Check GitHub Actions logs**
Look for specific error messages in the deployment logs to identify other issues.

### **Option 3: Manual deployment**
If GitHub Actions continues to fail, consider manual deployment through Azure Portal.

## 📞 **Support**

If you continue to have issues:
1. Check the GitHub Actions logs for specific error messages
2. Verify that all files are committed and pushed
3. Ensure the Azure App Service configuration is correct 