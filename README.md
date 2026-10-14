# Simen Vangen — Portfolio Website

Personal portfolio for Simen Vangen, a data scientist focused on embodied AI and robotics. Built as an interactive experience rather than a static CV site.

**Live site:** https://simenvangen.netlify.app  
**Backend API:** https://web-production-ad3a8.up.railway.app

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | Vanilla HTML/CSS/JS, deployed on Netlify |
| Backend | FastAPI, deployed on Railway |
| AI inference | Groq (LLaMA 3.3 70B) |
| RAG | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| Gesture detection | MediaPipe Hands (bundled locally) |

---

## Features

### 🎬 3D Project Carousel
Drum-style rotating carousel with lerp animation. Scroll, swipe, or use gesture control to navigate. Each card opens a full project overlay with description, media, and GitHub link.

### 👾 Welcome Animation
Types "Welcome to Simen Pofffolio", catches the typo, self-corrects, and fades into the portfolio.

### 🤖 RAG Chatbot
LLaMA 3.3 powered chatbot trained on Simen's CV, projects, thesis, and background. Uses FAISS vector search to retrieve relevant context before answering. Includes rate limiting, daily caps, and funny failsafe responses for nosy questions.

### 🔥 Bouncer AI (Roast Chat)
A savage AI that roasts visitors while making Simen look good. Opens as a chat window. Visitors can talk back. Powered by LLaMA 3.3 with a custom personality prompt.

### ✋ Gesture Navigation
MediaPipe Hands running in the browser via webcam. Bundled locally so it works without a CDN (China firewall workaround).

| Gesture | Action |
|---|---|
| ☝️ 1 finger | Previous project |
| ✌️ 2 fingers | Next project |
| 🖐 Open palm | Open chat |
| ✊ Fist | Close chat |

### 🎮 Robot Wars Game
Top-down wave survival game with adaptive enemy AI. Enemies escalate from chasing → dodging → strafing → flanking → predicting your movement. Features:
- 5 weapons with rarity tiers (Pistol 100% → Ray Gun 5%)
- Loot drops after every wave — pick 1 of 3
- Gold economy — collect drops, spend in upgrade shop
- 3 upgrade tiers per weapon
- Rocket AoE explosion, Ray Gun multi-beam
- Persistent leaderboard shared across all visitors

### 🏆 Leaderboard
"Leaderboard ↗" link in the top right corner. Shows top 10 all-time Robot Wars scores with player names, scores, and wave reached.

---

## File Structure

```
Portfolio web/
├── deploy/                         ← Netlify deploy folder
│   ├── index.html                  ← Main frontend file
│   ├── images/
│   │   ├── fruitwatch1.jpeg
│   │   ├── fruitwatch2.jpeg
│   │   └── lifepath.gif
│   └── mediapipe/                  ← Bundled locally (no CDN needed)
│       ├── hands.min.js
│       ├── camera_utils.min.js
│       ├── hands.binarypb
│       ├── hand_landmark_full.tflite
│       ├── hand_landmark_lite.tflite
│       ├── hands_solution_packed_assets.data
│       ├── hands_solution_packed_assets_loader.js
│       ├── hands_solution_simd_wasm_bin.js
│       └── hands_solution_simd_wasm_bin.wasm
├── scripts/
│   └── server.py                   ← FastAPI backend
├── data/                           ← RAG knowledge base
│   ├── cv.txt
│   ├── thesis.txt
│   ├── project_fruitwatch.txt
│   ├── project_life_path_simulator.txt
│   ├── interests_embodied_ai.txt
│   └── personal_story.txt
├── embeddings/
│   ├── index.faiss                 ← FAISS vector index
│   └── chunks.json                 ← Text chunks + metadata
├── logs/
│   └── requests.log
├── requirements.txt
├── Procfile                        ← web: python scripts/server.py
└── netlify.toml                    ← CORS + camera headers
```

---

## Backend API

Base URL: `https://web-production-ad3a8.up.railway.app`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/chat` | POST | RAG chatbot — 500 char limit, 10 req/hr per IP |
| `/roast` | POST | Bouncer AI opening roast |
| `/roast_reply` | POST | Bouncer AI reply to visitor message |
| `/stats` | GET | Usage dashboard |

### Rate limits
- 10 requests per IP per hour
- 200 requests per day total
- 500 character input limit on `/chat`
- `/roast` and `/roast_reply` bypass input limit

---

## Environment Variables

Set in Railway dashboard — never commit to repo.

```
GROQ_API_KEY=gsk_...
```

---

## Deployment

### Frontend (Netlify)
```bash
# Edit deploy/index.html
git add deploy/
git commit -m "update frontend"
git push
# Then drag deploy/ folder onto Netlify
```

### Backend (Railway)
```bash
# Edit scripts/server.py
git add scripts/server.py
git commit -m "update backend"
git push
# Railway auto-redeploys on push
```

### Rebuild RAG index
```bash
python scripts/build_index.py
```

---

## Netlify Headers

Required for MediaPipe WebAssembly and camera access over HTTPS:

```toml
[[headers]]
  for = "/*"
  [headers.values]
    Cross-Origin-Opener-Policy = "same-origin"
    Cross-Origin-Embedder-Policy = "require-corp"
    Cross-Origin-Resource-Policy = "cross-origin"
    Permissions-Policy = "camera=*"
```

---

## Projects

### FruitWatch
Computer vision platform for fruit monitoring. YOLOv8 ripeness detection + PlantVillage disease classifier (99.8% accuracy). Runs on Raspberry Pi 5 for edge inference.

### Life Path Simulator
AI life simulation game powered by a custom 88M parameter decoder-only transformer trained from scratch in PyTorch. 50,000 synthetic training examples. No external LLM APIs.

---

## Known Issues

- MediaPipe CDN is blocked in China — solved by bundling all model files locally in `deploy/mediapipe/`
- Gesture detection requires HTTPS (works on Netlify, use live server locally)
- Railway requires `GROQ_API_KEY` set as environment variable in dashboard

---

## Author

**Simen Vangen** — Data Scientist, Embodied AI & Robotics  
GitHub: [SimenVangen](https://github.com/SimenVangen)  
Site: [simenvangen.netlify.app](https://simenvangen.netlify.app)
