print("Hello from Python!")
print("Current working directory test")
import os
print(f"Working directory: {os.getcwd()}")
print("Files in current directory:")
for item in os.listdir('.'):
    print(f"  {item}")
