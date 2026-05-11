#!/usr/bin/env python3
"""
Simpel test script om vLLM te testen
"""
import subprocess
import time
import requests
from openai import OpenAI

def test_vllm():
    model_name = 'meta-llama/Llama-2-7b-hf'
    port = 8000
    
    print("="*70)
    print("VLLM TEST SCRIPT")
    print("="*70)
    
    # Start vLLM server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", "0.8"
    ]
    
    print(f"\n1. Starting vLLM server with model: {model_name}")
    print(f"   Command: {' '.join(cmd)}\n")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wacht tot server klaar is
    base_url = f"http://127.0.0.1:{port}/v1"
    max_retries = 60
    retry_count = 0
    
    print("   Waiting for server to start...")
    while retry_count < max_retries:
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                print(f"   ✓ Server is ready!\n")
                break
        except requests.exceptions.RequestException as e:
            pass
        
        retry_count += 1
        if retry_count % 10 == 0:
            print(f"   Still waiting... ({retry_count}/{max_retries})")
        time.sleep(2)
    
    if retry_count >= max_retries:
        print("   ❌ Server startup failed!")
        print("\n2. Checking error output...")
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2)
            if stderr:
                print("   Server errors:")
                print(stderr[:1000])
        except:
            pass
        return False
    
    # Test API
    print("2. Testing API connection...")
    try:
        client = OpenAI(api_key="dummy", base_url=base_url)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hello! What is 2+2?'}],
            max_tokens=50,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print(f"   ✓ API works!")
        print(f"   Model response: {result}\n")
        
        print("="*70)
        print("✓ VLLM TEST SUCCESSFUL!")
        print("="*70)
        
        # Stop server
        process.terminate()
        print("\nServer stopped.")
        return True
        
    except Exception as e:
        print(f"   ❌ API test failed: {e}\n")
        process.terminate()
        return False

if __name__ == "__main__":
    try:
        success = test_vllm()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
