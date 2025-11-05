import hopsworks
import os

api_key = os.environ.get("HOPSWORKS_API_KEY")
project_name = os.environ.get("HOPSWORKS_PROJECTcl")

print("🔑  Testing Hopsworks login...")
project = hopsworks.login(api_key_value=api_key, project=project_name)
print(f"✅  Logged in successfully to project: {project.name}")
