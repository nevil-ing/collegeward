import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

def get_ip_address():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        response.raise_for_status()
        ip_data = response.json()
        return ip_data.get('ip', 'IP address not found')
    except requests.RequestException as e:
        return f"Error occurred: {e}"

@app.get("/ip")
async def get_client_ip(request: Request):
    client_host = request.client.host
    return JSONResponse(content={"ipnewtest": get_ip_address()})