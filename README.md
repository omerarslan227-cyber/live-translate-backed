# BridgeCall Backend

FastAPI backend for BridgeCall realtime rooms, WebRTC signaling, chat, subtitle forwarding, Whisper speech-to-text, and DeepL translation.

## Run locally

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# fill DEEPL_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Flutter can point to this backend with:

```powershell
flutter run --dart-define=BRIDGECALL_WS_URL=ws://10.0.2.2:8000
```

Use `ws://localhost:8000` for desktop/web and your machine LAN IP for a physical iPhone/Android device.
