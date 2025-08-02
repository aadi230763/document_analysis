#!/usr/bin/env python3
"""
Force Azure App Service Configuration Update
This script uses Azure CLI to update the startup command
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔧 Force Azure App Service Configuration Update")
    print("=" * 50)
    
    # Check if Azure CLI is installed
    success, stdout, stderr = run_command("az --version")
    if not success:
        print("❌ Azure CLI not found. Please install it first:")
        print("   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return False
    
    print("✅ Azure CLI found")
    
    # Check if logged in
    success, stdout, stderr = run_command("az account show")
    if not success:
        print("❌ Not logged into Azure. Please run: az login")
        return False
    
    print("✅ Logged into Azure")
    
    # Get the resource group and app name from environment or user input
    resource_group = os.getenv("AZURE_RESOURCE_GROUP", "your-resource-group")
    app_name = os.getenv("AZURE_APP_NAME", "bajaj")
    
    if resource_group == "your-resource-group":
        resource_group = input("Enter your Azure Resource Group name: ")
    
    print(f"📋 Using Resource Group: {resource_group}")
    print(f"📋 Using App Service: {app_name}")
    
    # Update the startup command
    print("\n🔄 Updating startup command to 'bash startup.sh'...")
    
    command = f'az webapp config set --resource-group {resource_group} --name {app_name} --startup-file "bash startup.sh"'
    success, stdout, stderr = run_command(command)
    
    if success:
        print("✅ Startup command updated successfully!")
        print("\n🔄 Restarting the App Service...")
        
        # Restart the app service
        restart_command = f'az webapp restart --resource-group {resource_group} --name {app_name}'
        success, stdout, stderr = run_command(restart_command)
        
        if success:
            print("✅ App Service restarted successfully!")
            print("\n📋 Next steps:")
            print("1. Wait 2-3 minutes for the restart to complete")
            print("2. Check the Azure Portal Log Stream")
            print("3. Look for the success message:")
            print("   🚀 Starting LLM-Powered Document Analysis System on Azure")
            print("4. Test the endpoint with: python test_azure_endpoint.py")
            return True
        else:
            print(f"❌ Failed to restart App Service: {stderr}")
            return False
    else:
        print(f"❌ Failed to update startup command: {stderr}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 