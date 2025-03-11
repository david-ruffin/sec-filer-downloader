#!/bin/bash
# Simple deployment script for Azure App Service

# Create a deployment package without unnecessary files
echo "Creating deployment package..."
mkdir -p deploy
cp -r $(ls -A | grep -v "filings\|deploy\|.git\|.env\|__pycache__\|.pyc") deploy/

# Create the startup file if it doesn't exist
if [ ! -f deploy/startup.txt ]; then
  echo "gunicorn --bind=0.0.0.0 --timeout 600 app:app" > deploy/startup.txt
fi

# Create web.config for Azure
cat > deploy/web.config << EOL
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <appSettings>
    <add key="PYTHONPATH" value="%PYTHONPATH%;/home/site/wwwroot" />
  </appSettings>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified" />
    </handlers>
    <httpPlatform processPath="%HOME%\site\wwwroot\env\Scripts\python.exe" arguments="-m gunicorn --bind=0.0.0.0 --timeout 600 app:app" requestTimeout="00:04:00" />
  </system.webServer>
</configuration>
EOL

echo "Deployment package created. Use Azure CLI to deploy:"
echo "az webapp deployment source config-local-git --name secv1 --resource-group YOUR_RESOURCE_GROUP"
echo "cd deploy && git init && git add . && git commit -m 'Deploy'"
echo "git remote add azure [GIT_URL_FROM_PREVIOUS_COMMAND]"
echo "git push azure master"
