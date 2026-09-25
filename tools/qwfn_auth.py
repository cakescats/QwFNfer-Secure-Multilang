"""Accounts, sessions and the API key for the qwfn console.

State lives in the console's config under "auth" (the file is written 0600):

    {"users":    {"login": {"salt": hex, "hash": hex, "iter": N}},
     "sessions": {sha256(token): {"user": login, "exp": unix time}},
     "api_key":  "qwfn-...", "require_key": true, "server_host": "127.0.0.1"}

Passwords are PBKDF2-HMAC-SHA256 with a per-user salt; the session cookie carries a
random token and only its hash is stored, so the config file does not hand out live
sessions. Failed sign-ins are rate-limited per client address.
"""
import hashlib, hmac, re, secrets, threading, time

ITER = 600_000                 # OWASP 2023 figure for PBKDF2-HMAC-SHA256
MIN_PASSWORD = 8
SESSION_S = 12 * 3600
REMEMBER_S = 30 * 24 * 3600
LOGIN_RE = re.compile(r"^[\w.\-]{1,64}$")


def new_api_key():
    return "qwfn-" + secrets.token_urlsafe(32)


def _hash(password, salt, iters):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters).hex()


class Auth:
    def __init__(self, config, save):
        """`config` is the console's CONFIG dict, `save` persists it."""
        self.cfg, self.save = config, save
        a = config.setdefault("auth", {})
        a.setdefault("users", {}); a.setdefault("sessions", {})
        a.setdefault("require_key", True); a.setdefault("server_host", "127.0.0.1")
        if not a.get("api_key"): a["api_key"] = new_api_key(); save()
        self.lock = threading.Lock()
        self.fails = {}                     # ip -> [count, last failure time]
        self.setup_code = secrets.token_hex(4) if not a["users"] else None

    @property
    def a(self): return self.cfg["auth"]

    def has_users(self): return bool(self.a["users"])

    # ---- passwords ---------------------------------------------------------------------
    def _check_new(self, login, password):
        if not LOGIN_RE.match(login or ""): return "the login must be 1-64 letters, digits, dots, dashes or underscores"
        if len(password or "") < MIN_PASSWORD: return "the password must be at least %d characters" % MIN_PASSWORD
        return None

    def _set_password(self, login, password):
        salt = secrets.token_bytes(16)
        self.a["users"][login] = {"salt": salt.hex(), "hash": _hash(password, salt, ITER), "iter": ITER}

    def verify(self, login, password):
        u = self.a["users"].get(login or "")
        if not u:
            _hash(password or "", b"\0" * 16, ITER)        # same cost whether or not the login exists
            return False
        return hmac.compare_digest(u["hash"], _hash(password or "", bytes.fromhex(u["salt"]), int(u.get("iter", ITER))))

    def setup(self, login, password, from_local, code):
        with self.lock:
            if self.has_users(): return "an account already exists: sign in instead"
            if not from_local and (not code or not self.setup_code or not hmac.compare_digest(code, self.setup_code)):
                return "the first account can only be created from this machine, or with the setup code printed in the console's terminal" if not code else "the setup code is wrong"
            err = self._check_new(login, password)
            if err: return err
            self._set_password(login, password); self.setup_code = None; self.save()
            return None

    def add_user(self, login, password):
        with self.lock:
            if login in self.a["users"]: return "that login already exists"
            err = self._check_new(login, password)
            if err: return err
            self._set_password(login, password); self.save(); return None

    def remove_user(self, login, me):
        with self.lock:
            if login == me: return "you cannot remove your own account"
            if login not in self.a["users"]: return "no such account"
            del self.a["users"][login]
            self.a["sessions"] = {k: v for k, v in self.a["sessions"].items() if v.get("user") != login}
            self.save(); return None

    def change_password(self, login, old, new):
        if not self.verify(login, old): return "the current password is wrong"
        err = self._check_new(login, new)
        if err: return err
        with self.lock:
            self._set_password(login, new)
            # every other session of this user ends with the old password
            self.a["sessions"] = {k: v for k, v in self.a["sessions"].items() if v.get("user") != login}
            self.save(); return None

    # ---- rate limit ----------------------------------------------------------------------
    def wait_s(self, ip):
        """Seconds this address must wait before another attempt (0 = go ahead). Free for 5
        failures, then 15 s doubling per failure up to 15 min."""
        f = self.fails.get(ip)
        if not f or f[0] < 5: return 0
        return max(0, int(f[1] + min(900, 15 * 2 ** (f[0] - 5)) - time.time()))

    def failed(self, ip):
        f = self.fails.setdefault(ip, [0, 0.0]); f[0] += 1; f[1] = time.time()

    def succeeded(self, ip):
        self.fails.pop(ip, None)

    # ---- sessions ----------------------------------------------------------------------
    def new_session(self, login, remember):
        token = secrets.token_urlsafe(32)
        with self.lock:
            now = time.time()
            self.a["sessions"] = {k: v for k, v in self.a["sessions"].items() if v.get("exp", 0) > now}
            self.a["sessions"][hashlib.sha256(token.encode()).hexdigest()] = {"user": login, "exp": now + (REMEMBER_S if remember else SESSION_S)}
            self.save()
        return token, (REMEMBER_S if remember else None)

    def user_of(self, token):
        if not token: return None
        s = self.a["sessions"].get(hashlib.sha256(token.encode()).hexdigest())
        if not s or s.get("exp", 0) < time.time() or s.get("user") not in self.a["users"]: return None
        return s["user"]

    def end_session(self, token):
        if not token: return
        with self.lock:
            if self.a["sessions"].pop(hashlib.sha256(token.encode()).hexdigest(), None) is not None: self.save()
