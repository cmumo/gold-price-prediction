import os
import json
import asyncio
import threading
import random
import string
from datetime import datetime
from collections import deque
from typing import List, Deque
import websocket as ws_client
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)

# === Configuration ===
MAX_BUFFER = 30
price_buffer: Deque[float] = deque(maxlen=MAX_BUFFER)
model = LogisticRegression()
trained = False
clients: list[WebSocket] = []

# === ML Functions ===
def build_features(prices: List[float]) -> np.ndarray:
    arr = np.array(prices)
    returns = np.diff(arr) / arr[:-1]
    momentum = returns[-5:].mean() if len(returns) >= 5 else 0.0
    ma_short = arr[-5:].mean() if len(arr) >= 5 else arr.mean()
    ma_long = arr.mean()
    return np.array([[momentum, ma_short - ma_long]])

def train_model():
    global trained, model
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    trained = True
    logging.info("Model trained successfully")

def calculate_tp_sl(current_price: float, signal: str, confidence: float):
    if len(price_buffer) < 10:
        return None
    
    prices = list(price_buffer)
    volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices))) / (len(prices) - 1)
    multiplier = confidence / 50
    
    if signal == 'BUY':
        return {
            'tp1': round(current_price + (volatility * 1.5 * multiplier), 2),
            'tp2': round(current_price + (volatility * 3 * multiplier), 2),
            'tp3': round(current_price + (volatility * 5 * multiplier), 2),
            'sl': round(current_price - (volatility * 2 * multiplier), 2)
        }
    elif signal == 'SELL':
        return {
            'tp1': round(current_price - (volatility * 1.5 * multiplier), 2),
            'tp2': round(current_price - (volatility * 3 * multiplier), 2),
            'tp3': round(current_price - (volatility * 5 * multiplier), 2),
            'sl': round(current_price + (volatility * 2 * multiplier), 2)
        }
    return None

def predict_signal() -> dict:
    if len(price_buffer) < 10 or not trained:
        return {"signal": "HOLD", "direction": "HOLD", "confidence": 50, "levels": None}
    
    X = build_features(list(price_buffer))
    prob = model.predict_proba(X)[0][1]
    confidence = abs(prob - 0.5) * 200
    
    if prob > 0.60:
        signal = "BUY"
        direction = "UP"
    elif prob < 0.40:
        signal = "SELL"
        direction = "DOWN"
    else:
        signal = "HOLD"
        direction = "HOLD"
    
    levels = None
    if signal != "HOLD" and len(price_buffer) > 0:
        levels = calculate_tp_sl(price_buffer[-1], signal, min(confidence, 100))
    
    return {
        "signal": signal,
        "direction": direction,
        "confidence": round(min(confidence, 100), 1),
        "levels": levels
    }

# === TradingView WebSocket Handler ===
async def broadcast_to_clients(payload: str):
    dead = []
    for ws in clients[:]:
        try:
            await ws.send_text(payload)
        except:
            dead.append(ws)
    for ws in dead:
        if ws in clients:
            clients.remove(ws)

def tradingview_websocket():
    def on_message(ws, message):
        try:
            if message.startswith('~m~'):
                parts = message.split('~m~')
                for part in parts:
                    if part and len(part) > 0 and part[0] == '{':
                        try:
                            data = json.loads(part)
                            if 'm' in data and data['m'] == 'qsd':
                                if 'p' in data and len(data['p']) > 1:
                                    quote_data = data['p'][1]
                                    if 'v' in quote_data and 'lp' in quote_data['v']:
                                        price = float(quote_data['v']['lp'])
                                        
                                        price_buffer.append(price)
                                        prediction = predict_signal()
                                        
                                        payload = json.dumps({
                                            "price": price,
                                            "signal": prediction["signal"],
                                            "direction": prediction["direction"],
                                            "confidence": prediction["confidence"],
                                            "levels": prediction["levels"]
                                        })
                                        
                                        # Run async broadcast in new loop
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(broadcast_to_clients(payload))
                                        loop.close()
                                        
                                        logging.info(f"Price: ${price:.2f} | Signal: {prediction['signal']} | Clients: {len(clients)}")
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logging.error(f"Message error: {e}")
    
    def on_error(ws, error):
        logging.error(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        logging.warning("TradingView WebSocket closed, reconnecting in 5s...")
        import time
        time.sleep(5)
        tradingview_websocket()
    
    def on_open(ws):
        session = 'qs_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        
        ws.send('~m~' + str(len('~h~1')) + '~m~~h~1')
        
        locale_msg = '{"m":"set_locale","p":["en","US"]}'
        ws.send(f'~m~{len(locale_msg)}~m~{locale_msg}')
        
        quote_session = f'qs_{session}'
        create_session = json.dumps({"m": "quote_create_session", "p": [quote_session]})
        ws.send(f'~m~{len(create_session)}~m~{create_session}')
        
        add_symbols = json.dumps({"m": "quote_add_symbols", "p": [quote_session, "OANDA:XAUUSD"]})
        ws.send(f'~m~{len(add_symbols)}~m~{add_symbols}')
        
        logging.info("Connected to TradingView - Live gold prices streaming!")

    ws_app = ws_client.WebSocketApp(
        "wss://data.tradingview.com/socket.io/websocket",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws_app.run_forever()

# === FastAPI App ===
app = FastAPI()

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Gold Trading Terminal</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * { 
      margin: 0; 
      padding: 0; 
      box-sizing: border-box; 
    }
    
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
      color: #fff;
      min-height: 100vh;
      padding: 0.5rem;
      overflow-x: hidden;
    }
    
    .container { 
      max-width: 1600px; 
      margin: 0 auto;
      padding: 0 1rem;
    }

    /* Header */
    .header {
      text-align: center;
      padding: 1.5rem 0;
      margin-bottom: 1.5rem;
      position: relative;
    }
    
    .header h1 {
      font-size: clamp(1.5rem, 5vw, 3rem);
      font-weight: 900;
      background: linear-gradient(135deg, #fbbf24 0%, #fcd34d 50%, #f59e0b 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.5rem;
      text-shadow: 0 0 40px rgba(251, 191, 36, 0.3);
      letter-spacing: -0.02em;
    }
    
    .status {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: clamp(0.7rem, 2vw, 0.875rem);
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      padding: 0.5rem 1rem;
      background: rgba(148, 163, 184, 0.1);
      border-radius: 2rem;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .status-dot {
      width: 8px; 
      height: 8px; 
      border-radius: 50%; 
      background: #10b981;
      animation: pulse 2s infinite;
      box-shadow: 0 0 10px #10b981;
    }
    
    @keyframes pulse { 
      0%, 100% { opacity: 1; transform: scale(1); } 
      50% { opacity: 0.5; transform: scale(0.9); } 
    }

    /* Top Row Layout - DESKTOP: Chart 60% + Price/Signal 40% */
    .top-row {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    
    .right-col {
      display: grid;
      gap: 1rem;
    }

    /* Card Styling */
    .card {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(100, 116, 139, 0.3);
      border-radius: 1.25rem;
      padding: clamp(1rem, 3vw, 2rem);
      box-shadow: 
        0 20px 25px -5px rgba(0, 0, 0, 0.4),
        0 10px 10px -5px rgba(0, 0, 0, 0.2),
        inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
      position: relative;
      overflow: hidden;
      transition: all 0.3s ease;
    }
    
    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    
    .card:hover {
      transform: translateY(-2px);
      box-shadow: 
        0 25px 30px -5px rgba(0, 0, 0, 0.5),
        0 15px 15px -5px rgba(0, 0, 0, 0.3);
      border-color: rgba(100, 116, 139, 0.5);
    }

    /* Chart Wrapper */
    #chart-wrapper {
      min-height: 400px;
      height: 100%;
      border-radius: 0.75rem;
      overflow: hidden;
      background: rgba(0, 0, 0, 0.3);
    }

    /* Price Display */
    .price-display { 
      text-align: center;
      padding: 1rem 0;
    }
    
    .price-label { 
      font-size: clamp(0.7rem, 2vw, 0.75rem);
      color: #94a3b8; 
      text-transform: uppercase; 
      letter-spacing: 0.15em; 
      margin-bottom: 1rem;
      font-weight: 600;
    }
    
    .price-value { 
      font-size: clamp(2.5rem, 8vw, 4.5rem);
      font-weight: 900; 
      background: linear-gradient(135deg, #fbbf24, #f59e0b);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.03em;
      margin-bottom: 0.5rem;
      text-shadow: 0 0 40px rgba(251, 191, 36, 0.5);
      line-height: 1.1;
    }
    
    .price-direction { 
      font-size: clamp(1rem, 3vw, 1.25rem);
      color: #64748b;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    /* Signal Card */
    .signal-card { 
      text-align: center;
      position: relative;
      overflow: hidden;
      transition: all 0.4s ease;
    }
    
    .signal-card.buy { 
      background: linear-gradient(135deg, rgba(5, 150, 105, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
      border-color: #10b981;
      box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }
    
    .signal-card.sell { 
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
      border-color: #ef4444;
      box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }
    
    .signal-icon { 
      width: clamp(3rem, 10vw, 5rem);
      height: clamp(3rem, 10vw, 5rem);
      margin: 0 auto 1rem;
      filter: drop-shadow(0 0 10px currentColor);
      transition: transform 0.3s ease;
    }
    
    .signal-card:hover .signal-icon {
      transform: scale(1.1);
    }
    
    .signal-text { 
      font-size: clamp(2rem, 6vw, 3.5rem);
      font-weight: 900;
      margin-bottom: 0.5rem;
      letter-spacing: 0.05em;
      text-shadow: 0 0 20px currentColor;
    }
    
    .signal-text.buy { color: #10b981; }
    .signal-text.sell { color: #ef4444; }
    .signal-text.hold { color: #64748b; }
    
    .confidence { 
      font-size: clamp(1rem, 3vw, 1.25rem);
      color: rgba(255, 255, 255, 0.8);
      font-weight: 700;
    }

    /* Bottom Row */
    .bottom-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .levels-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
    }
    
    .levels-header h3 {
      font-size: clamp(1rem, 3vw, 1.25rem);
      font-weight: 700;
      color: #e2e8f0;
    }
    
    .level-item {
      padding: clamp(0.75rem, 2vw, 1rem);
      margin-bottom: 0.75rem;
      border-radius: 0.75rem;
      border-left: 4px solid;
      transition: all 0.3s ease;
      background: rgba(0, 0, 0, 0.2);
    }
    
    .level-item:hover { 
      transform: translateX(6px);
      background: rgba(0, 0, 0, 0.4);
    }
    
    .level-item.tp { 
      border-color: #10b981;
      background: linear-gradient(90deg, rgba(16, 185, 129, 0.1), transparent);
    }
    
    .level-item.sl { 
      border-color: #ef4444;
      background: linear-gradient(90deg, rgba(239, 68, 68, 0.1), transparent);
    }
    
    .level-row { 
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.25rem;
    }
    
    .level-label { 
      font-size: clamp(0.75rem, 2vw, 0.875rem);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    .level-label.tp { color: #10b981; }
    .level-label.sl { color: #ef4444; }
    
    .level-value { 
      font-size: clamp(1.25rem, 3vw, 1.75rem);
      font-weight: 900;
    }
    
    .level-value.tp { 
      color: #34d399;
      text-shadow: 0 0 10px rgba(52, 211, 153, 0.5);
    }
    
    .level-value.sl { 
      color: #f87171;
      text-shadow: 0 0 10px rgba(248, 113, 113, 0.5);
    }
    
    .level-desc { 
      font-size: clamp(0.65rem, 1.8vw, 0.75rem);
      color: rgba(255, 255, 255, 0.4);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    .no-signal { 
      text-align: center;
      padding: clamp(2rem, 5vw, 3rem) 0;
      color: #64748b;
      font-size: clamp(0.9rem, 2.5vw, 1.1rem);
    }

    .history-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: clamp(0.75rem, 2vw, 1rem);
      margin-bottom: 0.5rem;
      background: linear-gradient(90deg, rgba(15, 23, 42, 0.6), rgba(30, 41, 59, 0.4));
      border: 1px solid rgba(71, 85, 105, 0.3);
      border-radius: 0.75rem;
      transition: all 0.3s ease;
    }
    
    .history-item:hover { 
      background: linear-gradient(90deg, rgba(30, 41, 59, 0.6), rgba(51, 65, 85, 0.4));
      border-color: rgba(100, 116, 139, 0.5);
      transform: translateX(4px);
    }
    
    .history-time { 
      font-size: clamp(0.75rem, 2vw, 0.875rem);
      color: #94a3b8;
      font-family: 'Courier New', monospace;
      font-weight: 600;
    }
    
    .history-price { 
      font-size: clamp(1rem, 2.5vw, 1.25rem);
      font-weight: 700;
      background: linear-gradient(135deg, #fbbf24, #f59e0b);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .disclaimer {
      background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.1));
      border: 1px solid rgba(251, 191, 36, 0.3);
      border-radius: 0.75rem;
      padding: clamp(0.75rem, 2vw, 1rem);
      font-size: clamp(0.7rem, 2vw, 0.75rem);
      color: #fcd34d;
      line-height: 1.6;
    }

    /* Tablet Responsive - Keep desktop layout but adjust */
    @media (max-width: 1024px) {
      .container {
        padding: 0 0.75rem;
      }
    }
    
    /* MOBILE ONLY - Complete layout change */
    @media (max-width: 640px) {
      body {
        padding: 1rem;
      }
      
      .container {
        padding: 0 0.5rem;
      }
      
      .header {
        padding: 0.75rem 0;
        margin-bottom: 0.75rem;
      }
      
      .header h1 {
        font-size: 2rem;
      }
      
      .status {
        font-size: 0.7rem;
        padding: 0.4rem 0.75rem;
      }
      
      /* MOBILE: All single column */
      .top-row {
        grid-template-columns: 1fr;
        gap: 0.75rem;
      }
      
      .right-col {
        grid-template-columns: 1fr;
        gap: 0.75rem;
      }
      
      .bottom-row {
        grid-template-columns: 1fr;
        gap: 0.75rem;
      }
      
      /* Chart smaller on mobile */
      #chart-wrapper {
        min-height: 330px;
      }
      
      /* Compact cards */
      .card {
        padding: 0.875rem;
        border-radius: 1rem;
      }
      
      /* Smaller price display */
      .price-value {
        font-size: 2.75rem;
      }
      
      /* Smaller signal */
      .signal-icon {
        width: 3.5rem;
        height: 3.5rem;
        margin-bottom: 0.75rem;
      }
      
      .signal-text {
        font-size: 2.25rem;
      }
      
      .confidence {
        font-size: 1rem;
      }
      
      /* Compact levels */
      .levels-header {
        margin-bottom: 1rem;
      }
      
      .levels-header h3 {
        font-size: 0.95rem;
      }
      
      .level-item {
        padding: 0.65rem;
        margin-bottom: 0.6rem;
      }
      
      .level-label {
        font-size: 0.75rem;
      }
      
      .level-value {
        font-size: 1.25rem;
      }
      
      .level-desc {
        font-size: 0.65rem;
      }
      
      /* Compact history */
      .history-item {
        padding: 0.65rem;
        margin-bottom: 0.5rem;
      }
      
      .history-time {
        font-size: 0.75rem;
      }
      
      .history-price {
        font-size: 1rem;
      }
      
      /* Smaller disclaimer */
      .disclaimer {
        padding: 0.75rem;
        font-size: 0.7rem;
        margin-top: 0.75rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>⚡ GOLD PRICE PREDICTION</h1>
      <div class="status">
        <span class="status-dot"></span>
        <span id="status">Connecting...</span>
      </div>
    </div>

    <!-- Top Row: Chart + Price/Signal -->
    <div class="top-row">
      <!-- Chart -->
      <div class="card">
        <div id="chart-wrapper">
          <div class="tradingview-widget-container" style="width:100%;height:100%;">
            <div id="tradingview_chart" style="width:100%;height:100%;"></div>
          </div>
        </div>
      </div>

      <!-- Price + Signal -->
      <div class="right-col">
        <!-- Live Price -->
        <div class="card">
          <div class="price-display">
            <div class="price-label">XAU/USD Live Price</div>
            <div class="price-value" id="price">—</div>
            <div class="price-direction">
              <span id="direction">—</span>
            </div>
          </div>
        </div>

        <!-- Signal -->
        <div class="card signal-card" id="signalCard">
          <svg class="signal-icon" id="signalIcon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"></path>
          </svg>
          <div class="signal-text" id="signal">HOLD</div>
          <div class="confidence" id="confidence"></div>
        </div>
      </div>
    </div>

    <!-- Bottom Row: Levels + Recent Prices -->
    <div class="bottom-row">
      <!-- TP / SL Levels -->
      <div class="card">
        <div class="levels-header">
          <svg width="20" height="20" fill="none" stroke="#60a5fa" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" stroke-width="2"></circle>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6l4 2"></path>
          </svg>
          <h3>Take Profit & Stop Loss</h3>
        </div>
        <div id="levels"></div>
      </div>

      <!-- Recent Prices -->
      <div class="card">
        <div class="levels-header">
          <svg width="20" height="20" fill="none" stroke="#fbbf24" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
          </svg>
          <h3>Recent Prices</h3>
        </div>
        <div id="history"></div>
      </div>
    </div>

    <div class="disclaimer">
      <strong>⚠️ Risk Warning:</strong> Trading gold and financial instruments involves substantial risk. 
      This system is for educational purposes only. Always conduct your own research and consult with licensed financial advisors before making any trading decisions.
    </div>
  </div>

  <!-- TradingView Widget -->
  <script type="text/javascript">
    window.addEventListener("load", function() {
      if (window.innerWidth > 640) {
        new TradingView.widget({
          "container_id": "tradingview_chart",
          "width": "100%",
          "height": "100%",
          "symbol": "OANDA:XAUUSD",
          "interval": "15",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#0f172a",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "hide_side_toolbar": false,
          "autosize": true
        });
      } else {
        // Simplified chart for mobile
        new TradingView.widget({
          "container_id": "tradingview_chart",
          "width": "100%",
          "height": "100%",
          "symbol": "OANDA:XAUUSD",
          "interval": "15",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#0f172a",
          "enable_publishing": false,
          "allow_symbol_change": false,
          "hide_side_toolbar": true,
          "hide_top_toolbar": true,
          "autosize": true
        });
      }
    });
  </script>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>

  <!-- WebSocket Logic -->
  <script>
    // Auto-detect correct WebSocket protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
    const priceHistory = [];

    const upIcon = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>';
    const downIcon = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"></path>';
    const holdIcon = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14"></path>';

    ws.onopen = () => {
      console.log('WebSocket connected!');
      document.getElementById('status').textContent = 'Live';
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      console.log('Received data:', data); // Debug log

      document.getElementById('price').textContent = `$${data.price.toFixed(2)}`;
      document.getElementById('direction').textContent = data.direction;

      const signalEl = document.getElementById('signal');
      const signalCard = document.getElementById('signalCard');
      const signalIcon = document.getElementById('signalIcon');
      const confidenceEl = document.getElementById('confidence');

      signalEl.textContent = data.signal;
      signalEl.className = `signal-text ${data.signal.toLowerCase()}`;
      signalCard.className = `card signal-card ${data.signal.toLowerCase()}`;

      if (data.signal === 'BUY') {
        signalIcon.innerHTML = upIcon;
        signalIcon.style.color = '#10b981';
      } else if (data.signal === 'SELL') {
        signalIcon.innerHTML = downIcon;
        signalIcon.style.color = '#ef4444';
      } else {
        signalIcon.innerHTML = holdIcon;
        signalIcon.style.color = '#64748b';
      }

      confidenceEl.textContent = data.signal !== 'HOLD' ? `${data.confidence}% Confidence` : '';

      const levelsEl = document.getElementById('levels');
      if (data.levels) {
        levelsEl.innerHTML = `
          <div class="level-item tp">
            <div class="level-row">
              <span class="level-label tp">TP1</span>
              <span class="level-value tp">$${data.levels.tp1}</span>
            </div>
            <div class="level-desc">Conservative Target</div>
          </div>
          <div class="level-item tp">
            <div class="level-row">
              <span class="level-label tp">TP2</span>
              <span class="level-value tp">$${data.levels.tp2}</span>
            </div>
            <div class="level-desc">Moderate Target</div>
          </div>
          <div class="level-item tp">
            <div class="level-row">
              <span class="level-label tp">TP3</span>
              <span class="level-value tp">$${data.levels.tp3}</span>
            </div>
            <div class="level-desc">Aggressive Target</div>
          </div>
          <div class="level-item sl">
            <div class="level-row">
              <span class="level-label sl">STOP LOSS</span>
              <span class="level-value sl">$${data.levels.sl}</span>
            </div>
            <div class="level-desc">Risk Management</div>
          </div>
        `;
      } else {
        levelsEl.innerHTML = '<div class="no-signal">No Active Signal<br><small>Waiting for opportunity...</small></div>';
      }

      priceHistory.push({ time: new Date(), price: data.price });
      if (priceHistory.length > 6) priceHistory.shift();

      const historyEl = document.getElementById('history');
      historyEl.innerHTML = priceHistory.slice().reverse().map(h => `
        <div class="history-item">
          <span class="history-time">${h.time.toLocaleTimeString()}</span>
          <span class="history-price">$${h.price.toFixed(2)}</span>
        </div>
      `).join('');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      document.getElementById('status').textContent = 'Reconnecting...';
      setTimeout(() => location.reload(), 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      document.getElementById('status').textContent = 'Connection Error';
    };
    
    // Prevent zoom on double tap for mobile
    let lastTouchEnd = 0;
    document.addEventListener('touchend', (event) => {
      const now = Date.now();
      if (now - lastTouchEnd <= 300) {
        event.preventDefault();
      }
      lastTouchEnd = now;
    }, false);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    logging.info(f"Client connected. Total: {len(clients)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)
        logging.info(f"Client disconnected. Total: {len(clients)}")

@app.on_event("startup")
async def startup():
    train_model()
    # Start TradingView WebSocket in background thread
    tv_thread = threading.Thread(target=tradingview_websocket, daemon=True)
    tv_thread.start()
    logging.info("Gold Trading Terminal started with TradingView real-time data!")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
