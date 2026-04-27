#!/usr/bin/env python3
"""
ChatGPT HTTP Helper — curl_cffi Session-based
Bypasses Cloudflare via Safari TLS fingerprint impersonation.
Uses Session object for automatic cookie jar management — every Set-Cookie
from chatgpt.com is captured and reused on subsequent requests.

Runs as a lightweight HTTP server on port 1436.
The TypeScript provider calls this for all chatgpt.com requests.
"""

import json, sys, os, time, hashlib, base64, uuid, signal, random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
from typing import Any

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    print("[http-helper] ERROR: curl_cffi not installed. Run: pip3 install curl_cffi", flush=True)
    sys.exit(1)

PORT = int(os.environ.get('CHATGPT_BRIDGE_PORT', '1436'))
IMPERSONATE = "safari17_0"
CHATGPT_BASE = "https://chatgpt.com"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(SCRIPT_DIR, '.cookies.json')

# ── State ──
access_token = None
token_expires_at = 0
device_id = str(uuid.uuid4())
cached_dpl = ""
cached_scripts = []
dpl_fetched_at = 0
lock = threading.Lock()
cookie_update_count = 0  # Track how many times cookies got updated
last_cookie_update = 0

# ── Session (the key change — replaces raw cffi_requests calls) ──
session: Any = None

# ── Cookie management ──
def load_cookies_from_env():
    """Load initial cookies from env var or .env file"""
    cookies_str = os.environ.get('CHATGPT_COOKIES', '')
    if not cookies_str:
        env_file = os.path.join(SCRIPT_DIR, '.env')
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith('CHATGPT_COOKIES='):
                        cookies_str = line.split('=', 1)[1].strip()
                        break
    
    cookies = {}
    for part in cookies_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            cookies[k.strip()] = v.strip()
    return cookies

def load_persisted_cookies():
    """Load cookies from .cookies.json if it exists and is fresh (< 24h)"""
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE) as f:
                data = json.load(f)
            saved_at = data.get('_saved_at', 0)
            # Use persisted cookies if less than 24 hours old
            if time.time() - saved_at < 86400:
                cookies = {k: v for k, v in data.items() if k != '_saved_at'}
                if cookies:
                    print(f"[http-helper] Loaded {len(cookies)} persisted cookies (age: {int((time.time() - saved_at) / 60)}min)", flush=True)
                    return cookies
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def save_cookies():
    """Persist current session cookies to .cookies.json"""
    global cookie_update_count, last_cookie_update
    if session is None:
        return
    try:
        cookies_dict: dict[str, object] = dict(session.cookies)
        cookies_dict['_saved_at'] = time.time()
        with open(COOKIE_FILE, 'w') as f:
            json.dump(cookies_dict, f, indent=2)
        cookie_update_count += 1
        last_cookie_update = time.time()
    except IOError as e:
        print(f"[http-helper] Cookie save error: {e}", flush=True)

def get_cookie_count():
    """Get current number of cookies in session"""
    if session is None:
        return 0
    return len(dict(session.cookies))

def init_session():
    """Create Session and load cookies (persisted first, then .env as fallback)"""
    global session
    
    session = cffi_requests.Session(impersonate=IMPERSONATE)
    
    # Layer 1: Load from .env (base cookies including session token)
    env_cookies = load_cookies_from_env()
    if env_cookies:
        session.cookies.update(env_cookies)
        print(f"[http-helper] Loaded {len(env_cookies)} cookies from .env", flush=True)
    
    # Layer 2: Override with persisted cookies (fresher CF cookies)
    persisted = load_persisted_cookies()
    if persisted:
        session.cookies.update(persisted)
        print(f"[http-helper] Merged {len(persisted)} persisted cookies on top", flush=True)
    
    total = get_cookie_count()
    print(f"[http-helper] Session initialized with {total} cookies total", flush=True)

def refresh_cloudflare_cookies():
    """Visit homepage to get fresh Cloudflare cookies"""
    global cached_dpl, cached_scripts, dpl_fetched_at
    import re
    
    print("[http-helper] Refreshing Cloudflare cookies via homepage...", flush=True)
    try:
        r = session.get(
            CHATGPT_BASE,
            headers={**base_headers(), 'Accept': 'text/html'},
            timeout=15,
        )
        
        if r.status_code == 200:
            # Also extract DPL while we're here
            html = r.text
            scripts = re.findall(r'src="(https://cdn\.oaistatic\.com/[^"]+\.js)"', html)
            if scripts:
                cached_scripts = scripts
            
            for s in scripts:
                m = re.search(r'(?:c/|_next/static/)([a-zA-Z0-9_-]+)/', s)
                if m:
                    cached_dpl = m.group(1)
                    break
            
            if not cached_dpl:
                m = re.search(r'data-build="([^"]+)"', html)
                if m:
                    cached_dpl = m.group(1)
            
            dpl_fetched_at = time.time()
            save_cookies()
            print(f"[http-helper] CF cookies refreshed ✓ (total: {get_cookie_count()})", flush=True)
            return True
        else:
            print(f"[http-helper] Homepage returned {r.status_code}", flush=True)
            return False
    except Exception as e:
        print(f"[http-helper] CF refresh error: {e}", flush=True)
        return False

def base_headers():
    return {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': CHATGPT_BASE,
        'Referer': f'{CHATGPT_BASE}/',
        'Oai-Device-Id': device_id,
        'Oai-Language': 'en-US',
    }

# ── Auth ──
def refresh_access_token():
    global access_token, token_expires_at
    now = time.time()
    if access_token and now < token_expires_at:
        return access_token
    
    print("[http-helper] Fetching access token...", flush=True)
    r = session.get(
        f'{CHATGPT_BASE}/api/auth/session',
        headers=base_headers(),
        timeout=15,
    )
    
    if r.status_code == 403:
        # CF cookies stale — refresh and retry
        print("[http-helper] 403 on auth/session — refreshing CF cookies...", flush=True)
        refresh_cloudflare_cookies()
        r = session.get(
            f'{CHATGPT_BASE}/api/auth/session',
            headers=base_headers(),
            timeout=15,
        )
    
    if r.status_code != 200:
        raise Exception(f"Session fetch failed: {r.status_code} - {r.text[:200]}")
    
    # Save cookies after successful auth (response may contain updated cookies)
    save_cookies()
    
    data = r.json()
    if 'accessToken' not in data:
        raise Exception(f"No accessToken in response: {json.dumps(data)[:200]}")
    
    access_token = data['accessToken']
    token_expires_at = now + 20 * 60  # Refresh every 20 min
    print(f"[http-helper] Access token obtained ✓ (len={len(access_token)})", flush=True)
    return access_token

# ── DPL extraction ──
def refresh_dpl():
    global cached_dpl, cached_scripts, dpl_fetched_at
    import re
    
    now = time.time()
    if cached_dpl and now - dpl_fetched_at < 15 * 60:
        return
    
    # refresh_cloudflare_cookies() already extracts DPL, so just call that
    print("[http-helper] Fetching DPL from homepage...", flush=True)
    refresh_cloudflare_cookies()
    
    if not cached_dpl:
        cached_dpl = f"dpl-{uuid.uuid4().hex[:8]}"
    
    print(f"[http-helper] DPL: {cached_dpl}, scripts: {len(cached_scripts)}", flush=True)

# ── Proof of Work ──
NAVIGATOR_KEYS = [
    "hardwareConcurrency−16", "vendor−Google Inc.", "appVersion−5.0",
    "platform−Linux x86_64", "language−en-US", "onLine−true",
    "cookieEnabled−true", "deviceMemory−8", "maxTouchPoints−0",
]
DOCUMENT_KEYS = ["_reactListeningo743lnnpvdg", "location"]
WINDOW_KEYS = [
    "Object", "Function", "Array", "Number", "parseFloat", "parseInt",
    "Infinity", "NaN", "undefined", "Boolean", "String", "Symbol",
    "Date", "Promise", "RegExp", "Error", "JSON", "Math", "Intl",
    "ArrayBuffer", "Map", "Set", "WeakMap", "WeakSet", "Proxy",
    "Reflect", "console", "fetch", "crypto", "performance",
    "localStorage", "sessionStorage", "navigator", "location",
    "history", "screen", "document", "window", "self",
    "__NEXT_DATA__", "__next_f",
]
CORES = [8, 12, 16, 24, 32]
SCREEN_SUMS = [3000, 4000, 3120, 4160]

def get_parse_time():
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=-5)))
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{days[now.weekday()]} {months[now.month-1]} {now.day:02d} {now.year} {now.hour:02d}:{now.minute:02d}:{now.second:02d} GMT-0500 (Eastern Standard Time)"

def get_config():
    return [
        random.choice(SCREEN_SUMS),
        get_parse_time(),
        4294705152,
        0,  # nonce placeholder
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        random.choice(cached_scripts) if cached_scripts else "",
        cached_dpl,
        "en-US",
        "en-US,es-US,en,es",
        0,  # nonce >> 1 placeholder
        random.choice(NAVIGATOR_KEYS),
        random.choice(DOCUMENT_KEYS),
        random.choice(WINDOW_KEYS),
        time.perf_counter() * 1000,
        str(uuid.uuid4()),
        "",
        random.choice(CORES),
        time.time() * 1000 - (time.perf_counter() * 1000),
    ]

def solve_pow(seed, difficulty):
    diff_len = len(difficulty) // 2
    target = bytes.fromhex(difficulty)
    seed_bytes = seed.encode()
    config = get_config()
    
    # Pre-compute static parts
    part1 = json.dumps(config[:3], separators=(',', ':'))
    part2 = json.dumps(config[4:9], separators=(',', ':'))
    part3 = json.dumps(config[10:], separators=(',', ':'))
    
    for i in range(500000):
        config_json = f"{part1[:-1]},{i},{part2[1:-1]},{i >> 1},{part3[1:]}"
        b64 = base64.b64encode(config_json.encode()).decode()
        hash_val = hashlib.sha3_512(seed_bytes + b64.encode()).digest()
        
        if hash_val[:diff_len] <= target[:diff_len]:
            return f"gAAAAAB{b64}"
    
    # Fallback
    return "gAAAAA...xZ4D" + base64.b64encode(f'"{seed}"'.encode()).decode()

def generate_requirements_token():
    config = get_config()
    b64 = base64.b64encode(json.dumps(config).encode()).decode()
    return f"gAAAAAC{b64}"

# ── Sentinel ──
def get_chat_requirements(token):
    p_token = generate_requirements_token()
    
    r = session.post(
        f'{CHATGPT_BASE}/backend-api/sentinel/chat-requirements',
        headers={
            **base_headers(),
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        json={'p': p_token},
        timeout=15,
    )
    
    if r.status_code == 403:
        # CF cookies stale — refresh and retry
        print("[http-helper] 403 on sentinel — refreshing CF cookies...", flush=True)
        refresh_cloudflare_cookies()
        r = session.post(
            f'{CHATGPT_BASE}/backend-api/sentinel/chat-requirements',
            headers={
                **base_headers(),
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
            json={'p': p_token},
            timeout=15,
        )
    
    if r.status_code != 200:
        raise Exception(f"Sentinel failed: {r.status_code} - {r.text[:300]}")
    
    # Save cookies after sentinel (CF may have rotated)
    save_cookies()
    
    return r.json()

# ── Conversation ──
def send_conversation(token, model_slug, messages, sentinel_token, proof_token,
                      conversation_id=None, parent_message_id=None):
    chatgpt_messages = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if isinstance(content, list):
            text_parts = [p.get('text', '') for p in content if p.get('type') == 'text']
            content = '\n'.join(text_parts) if text_parts else json.dumps(content)
        
        chatgpt_messages.append({
            'id': str(uuid.uuid4()),
            'author': {'role': role},
            'create_time': time.time(),
            'content': {
                'content_type': 'text',
                'parts': [content if isinstance(content, str) else json.dumps(content)]
            },
            'metadata': {
                'developer_mode_connector_ids': [],
                'selected_connector_ids': [],
                'selected_sync_knowledge_store_ids': [],
                'selected_sources': [],
                'selected_github_repos': [],
                'selected_all_github_repos': False,
                'serialization_metadata': {'custom_symbol_offsets': []},
            }
        })
    
    body = {
        'action': 'next',
        'messages': chatgpt_messages,
        'parent_message_id': parent_message_id or str(uuid.uuid4()),
        'model': model_slug,
        'client_prepare_state': 'success',
        'timezone_offset_min': -420,
        'timezone': 'Asia/Jakarta',
        'conversation_mode': {'kind': 'primary_assistant'},
        'enable_message_followups': True,
        'system_hints': [],
        'supports_buffering': True,
        'supported_encodings': ['v1'],
        'client_contextual_info': {
            'is_dark_mode': True,
            'time_since_loaded': random.randint(500, 5000),
            'page_height': 1129,
            'page_width': 382,
            'pixel_ratio': 2,
            'screen_height': 1440,
            'screen_width': 2560,
            'app_name': 'chatgpt.com',
        },
        'paragen_cot_summary_display_override': 'allow',
        'force_parallel_switch': 'auto',
        'history_and_training_disabled': True,
    }

    if conversation_id:
        body['conversation_id'] = conversation_id
    
    # Add thinking_effort for thinking/pro models
    if 'thinking' in model_slug or 'pro' in model_slug or 'cot' in model_slug:
        body['thinking_effort'] = 'standard'
    
    headers = {
        **base_headers(),
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
    }
    
    if sentinel_token:
        headers['Openai-Sentinel-Chat-Requirements-Token'] = sentinel_token
    if proof_token:
        headers['Openai-Sentinel-Proof-Token'] = proof_token
    
    r = session.post(
        f'{CHATGPT_BASE}/backend-api/conversation',
        headers=headers,
        json=body,
        timeout=180,
    )
    
    if r.status_code == 403:
        # CF cookies stale mid-conversation — refresh and full retry
        print("[http-helper] 403 on conversation — refreshing CF cookies & retrying...", flush=True)
        refresh_cloudflare_cookies()
        
        # Re-auth + re-sentinel since cookies changed
        new_token = refresh_access_token()
        headers['Authorization'] = f'Bearer {new_token}'
        
        # Re-do sentinel
        try:
            requirements = get_chat_requirements(new_token)
            new_sentinel = requirements.get('token', '')
            if new_sentinel:
                headers['Openai-Sentinel-Chat-Requirements-Token'] = new_sentinel
            
            pow_data = requirements.get('proofofwork', {})
            if pow_data.get('required') and pow_data.get('seed') and pow_data.get('difficulty'):
                new_proof = solve_pow(pow_data['seed'], pow_data['difficulty'])
                headers['Openai-Sentinel-Proof-Token'] = new_proof
        except Exception as e:
            print(f"[http-helper] Re-sentinel failed: {e}", flush=True)
        
        r = session.post(
            f'{CHATGPT_BASE}/backend-api/conversation',
            headers=headers,
            json=body,
            timeout=180,
        )
    
    # Save cookies after conversation (CF rotates frequently)
    save_cookies()
    
    if r.status_code != 200:
        error_text = r.text[:500] if hasattr(r, 'text') else "Unknown error"
        return {'error': True, 'status': r.status_code, 'body': error_text}
    
    # Parse SSE stream (supports v1 delta encoding)
    full_text = ""
    response_conversation_id = ""
    last_event = ""
    last_path = ""  # Track last path for v1 delta encoding
    is_assistant_msg = False
    assistant_message_id = ""
    
    for line in r.text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Track event type
        if line.startswith('event: '):
            last_event = line[7:].strip()
            continue
        
        if not line.startswith('data: '):
            continue
        data = line[6:].strip()
        if data == '[DONE]':
            continue
        
        try:
            parsed = json.loads(data)
            
            # Extract conversation_id from various message types
            if isinstance(parsed, dict) and parsed.get('conversation_id'):
                response_conversation_id = parsed['conversation_id']
            
            # Handle type-based messages
            msg_type = parsed.get('type', '') if isinstance(parsed, dict) else ''
            
            # Full message object (initial delta with full message structure)
            if isinstance(parsed, dict) and 'v' in parsed and isinstance(parsed['v'], dict):
                v = parsed['v']
                if 'message' in v:
                    msg = v['message']
                    role = msg.get('author', {}).get('role', '')
                    is_assistant_msg = (role == 'assistant')
                    if is_assistant_msg and msg.get('id'):
                        assistant_message_id = msg['id']
                    if is_assistant_msg:
                        parts = msg.get('content', {}).get('parts', [])
                        if parts and isinstance(parts[0], str):
                            full_text = parts[0]
            
            # v1 delta encoding: append operation
            if isinstance(parsed, dict) and parsed.get('o') == 'append':
                path = parsed.get('p', last_path)
                if path:
                    last_path = path
                if path == '/message/content/parts/0' and is_assistant_msg:
                    full_text += str(parsed.get('v', ''))
            
            # v1 delta encoding: shorthand (just {"v": "text"} inheriting last path)
            if isinstance(parsed, dict) and 'v' in parsed and isinstance(parsed['v'], str) and 'o' not in parsed and 'p' not in parsed and 'type' not in parsed:
                if last_path == '/message/content/parts/0' and is_assistant_msg:
                    full_text += parsed['v']
            
            # Patch with batched ops
            if isinstance(parsed, dict) and parsed.get('o') == 'patch' and isinstance(parsed.get('v'), list):
                for op in parsed['v']:
                    if isinstance(op, dict):
                        if op.get('o') == 'append' and op.get('p') == '/message/content/parts/0':
                            full_text += str(op.get('v', ''))
            
            # Legacy format: direct message object
            if isinstance(parsed, dict) and 'message' in parsed and 'v' not in parsed:
                msg = parsed['message']
                if isinstance(msg, dict) and msg.get('author', {}).get('role') == 'assistant':
                    is_assistant_msg = True
                    if msg.get('id'):
                        assistant_message_id = msg['id']
                    parts = msg.get('content', {}).get('parts', [])
                    if parts and isinstance(parts[0], str):
                        full_text = parts[0]
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    
    # Hard delete conversation (not just hide)
    final_conv_id = response_conversation_id or conversation_id
    if final_conv_id and not conversation_id:
        try:
            headers = {
                **base_headers(),
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            }
            # Step 1: Archive it
            session.patch(
                f'{CHATGPT_BASE}/backend-api/conversation/{final_conv_id}',
                headers=headers,
                json={'is_archived': True},
                timeout=5,
            )
            # Step 2: Hard delete
            session.delete(
                f'{CHATGPT_BASE}/backend-api/conversation/{final_conv_id}',
                headers=headers,
                timeout=5,
            )
        except:
            pass
    
    return {
        'error': False,
        'text': full_text,
        'conversation_id': response_conversation_id or conversation_id or '',
        'message_id': assistant_message_id,
    }


class ChatGPTHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/health'):
            cookie_age = int(time.time() - last_cookie_update) if last_cookie_update else None
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'ok',
                'service': 'chatgpt-http-helper',
                'has_token': access_token is not None,
                'device_id': device_id,
                'impersonate': IMPERSONATE,
                'session_cookies': get_cookie_count(),
                'cookie_updates': cookie_update_count,
                'last_cookie_update_secs_ago': cookie_age,
            }).encode())
            return
        
        if path == '/cookies':
            # Debug endpoint: show current cookie names (not values)
            cookie_names = list(dict(session.cookies).keys()) if session else []
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'count': len(cookie_names),
                'names': cookie_names,
                'updates': cookie_update_count,
            }).encode())
            return
        
        if path == '/refresh':
            # Manual trigger: refresh CF cookies
            print("[http-helper] Manual cookie refresh triggered", flush=True)
            success = refresh_cloudflare_cookies()
            self.send_response(200 if success else 502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'refreshed': success,
                'cookies': get_cookie_count(),
            }).encode())
            return
        
        self.send_response(404)
        self.end_headers()
    
    def do_POST(self):
        path = urlparse(self.path).path

        if path == '/cleanup':
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length))
            conv_id = body.get('conversation_id')
            if conv_id:
                with lock:
                    try:
                        token = refresh_access_token()
                        headers = {
                            **base_headers(),
                            'Authorization': f'Bearer {token}',
                            'Content-Type': 'application/json',
                        }
                        # Archive then hard delete
                        session.patch(
                            f'{CHATGPT_BASE}/backend-api/conversation/{conv_id}',
                            headers=headers,
                            json={'is_archived': True},
                            timeout=5,
                        )
                        session.delete(
                            f'{CHATGPT_BASE}/backend-api/conversation/{conv_id}',
                            headers=headers,
                            timeout=5,
                        )
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'cleaned': True}).encode())
                    except Exception as e:
                        self.send_response(502)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'cleaned': False, 'error': str(e)}).encode())
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'conversation_id required'}).encode())
            return
        
        if path != '/chat':
            self.send_response(404)
            self.end_headers()
            return
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))
        
        model = body.get('model', 'auto')
        messages = body.get('messages', [])
        conv_id = body.get('conversation_id')
        parent_id = body.get('parent_message_id')
        
        print(f"[http-helper] → {model} ({len(messages)} messages)", flush=True)
        
        with lock:
            try:
                # 1. Get access token
                token = refresh_access_token()
                
                # 2. Get DPL
                refresh_dpl()
                
                # 3. Get sentinel requirements
                requirements = get_chat_requirements(token)
                sentinel_token = requirements.get('token', '')
                
                # 4. Solve PoW if needed
                proof_token = ""
                pow_data = requirements.get('proofofwork', {})
                if pow_data.get('required') and pow_data.get('seed') and pow_data.get('difficulty'):
                    start = time.time()
                    proof_token = solve_pow(pow_data['seed'], pow_data['difficulty'])
                    elapsed = (time.time() - start) * 1000
                    print(f"[http-helper] PoW solved in {elapsed:.0f}ms", flush=True)
                
                # 5. Send conversation
                result = send_conversation(
                    token, model, messages, sentinel_token, proof_token,
                    conversation_id=conv_id, parent_message_id=parent_id
                )
                
                if result.get('error'):
                    status_code = result.get('status', 500)
                    print(f"[http-helper] ✗ Error: {status_code} - {result.get('body', '')[:100]}", flush=True)
                    
                    # Invalidate token on auth errors
                    if status_code in (401, 403):
                        access_token = None
                        token_expires_at = 0
                    
                    self.send_response(502)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                else:
                    text_len = len(result.get('text', ''))
                    print(f"[http-helper] ✓ {model} → {text_len} chars", flush=True)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
            
            except Exception as e:
                print(f"[http-helper] ✗ Exception: {e}", flush=True)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': True, 'body': str(e)}).encode())


def main():
    global access_token, token_expires_at
    
    # Initialize session with cookies
    init_session()
    
    session_token = dict(session.cookies).get('__Secure-next-auth.session-token', '')
    if not session_token:
        print("[http-helper] ERROR: No session token found!", flush=True)
        sys.exit(1)
    
    print(f"[http-helper] Session token: {session_token[:30]}...", flush=True)
    print(f"[http-helper] Device ID: {device_id}", flush=True)
    print(f"[http-helper] Impersonate: {IMPERSONATE}", flush=True)
    
    # Pre-warm: visit homepage first (gets fresh CF cookies), then auth
    try:
        refresh_cloudflare_cookies()
        refresh_access_token()
    except Exception as e:
        print(f"[http-helper] Pre-warm failed: {e}", flush=True)
    
    # Allow port reuse to avoid "Address already in use" when restarted quickly
    import socket
    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()
    server = ReusableHTTPServer(('127.0.0.1', PORT), ChatGPTHandler)
    print(f"\n[http-helper] ChatGPT HTTP Helper ready on http://127.0.0.1:{PORT}", flush=True)
    print(f"[http-helper] POST /chat    — send messages", flush=True)
    print(f"[http-helper] POST /cleanup — delete conversation", flush=True)
    print(f"[http-helper] GET  /health  — status + cookie info", flush=True)
    print(f"[http-helper] GET  /cookies — list cookie names", flush=True)
    print(f"[http-helper] GET  /refresh — manual CF cookie refresh\n", flush=True)
    
    def shutdown(sig, frame):
        print("\n[http-helper] Shutting down...", flush=True)
        save_cookies()  # Persist cookies on shutdown
        server.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    server.serve_forever()

if __name__ == '__main__':
    main()
