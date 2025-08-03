#!/usr/bin/env python3
import sys
import os

# Write to both stdout and a file
output_file = "test_result.txt"

with open(output_file, "w") as f:
    f.write("Python execution test successful!\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Working directory: {os.getcwd()}\n")
    f.write("Files in directory:\n")
    for item in os.listdir('.'):
        f.write(f"  {item}\n")

print("Test completed - check test_result.txt")