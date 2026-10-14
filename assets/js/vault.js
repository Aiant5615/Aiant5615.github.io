/* Shared owner-only access for the tracker and LeetCode pages.
   The GitHub token is kept only as an AES-GCM ciphertext in localStorage ("vault"), locked with a password chosen by
   the owner (PBKDF2-SHA256, 200k iterations). The decrypted token lives in sessionStorage for the current tab only, so
   unlocking once covers every private page in that tab. Nothing here is sent anywhere except api.github.com. */
(function () {
  "use strict";
  const TOKEN_KEY = "gh_token", USER_KEY = "gh_user", VAULT_KEY = "gh_vault", SESSION_KEY = "gh_session";
  const lsGet = k => { try { return localStorage.getItem(k); } catch { return null; } };
  const lsSet = (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} };
  const ssGet = k => { try { return sessionStorage.getItem(k); } catch { return null; } };
  const ssSet = (k, v) => { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, v); } catch {} };
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const token = () => ssGet(SESSION_KEY) || "";
  const hasVault = () => !!lsGet(VAULT_KEY);
  const u8b64 = u => btoa(String.fromCharCode(...u)), b64u8 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  async function vaultKey(pw, salt) {
    const km = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw), "PBKDF2", false, ["deriveKey"]);
    return crypto.subtle.deriveKey({ name: "PBKDF2", salt, iterations: 200000, hash: "SHA-256" }, km, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
  }
  async function vaultSave(tok, pw) {
    const salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
    const ct = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv }, await vaultKey(pw, salt), new TextEncoder().encode(tok)));
    lsSet(VAULT_KEY, JSON.stringify({ v: 1, salt: u8b64(salt), iv: u8b64(iv), ct: u8b64(ct) }));
  }
  async function vaultOpen(pw) {   // → token, or null when the password is wrong
    let v; try { v = JSON.parse(lsGet(VAULT_KEY) || "null"); } catch { v = null; }
    if (!v || !v.ct) return null;
    try { return new TextDecoder().decode(await crypto.subtle.decrypt({ name: "AES-GCM", iv: b64u8(v.iv) }, await vaultKey(pw, b64u8(v.salt)), b64u8(v.ct))); }
    catch { return null; }
  }
  async function gh(path, opts = {}) {
    const r = await fetch(`https://api.github.com${path}`, { ...opts, cache: "no-store", headers: { Authorization: `Bearer ${token()}`, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", ...(opts.body ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) } });
    if (!r.ok) { let msg = `${r.status}`; try { msg += " " + ((await r.json()).message || ""); } catch {} const e = new Error(msg); e.status = r.status; throw e; }
    return r;
  }
  const ghJson = async (path, opts) => (await gh(path, opts)).json();
  // A file's text through the Contents API (works for private repos; raw.githubusercontent.com would not).
  const ghRaw = async (repo, path, branch = "main") => (await gh(`/repos/${repo}/contents/${path}?ref=${branch}`, { headers: { Accept: "application/vnd.github.raw+json" } })).text();

  // The lock UI. `repo` is checked for access when a token is first saved; `onChange()` runs after every lock /
  // unlock / reset so the page can reload or hide its data.
  function renderConnect(root, { repo, onChange = () => {} } = {}) {
    const TOKEN_RE = /^(github_pat_|ghp_|gho_)[A-Za-z0-9_]{20,}$/;
    const clean = s => String(s || "").replace(/[^\x21-\x7E]/g, "");   // printable ASCII only: strips spaces, "…", zero-width characters
    const legacy = lsGet(TOKEN_KEY);   // token saved in plain text by the earlier version: it is moved into the vault once a password is set
    const draw = (msg = "") => {
      const user = lsGet(USER_KEY), note = msg ? `· <span class="muted">${esc(msg)}</span>` : "";
      const pw = (id, ph) => `<input type="password" id="${id}" class="search" style="margin:0;flex:1;min-width:160px" placeholder="${ph}" autocomplete="${id === "gh-token" ? "off" : id === "gh-new" ? "new-password" : "current-password"}" aria-label="${ph}">`;
      if (token()) root.innerHTML = `<details class="more"><summary>🔓 Private · unlocked${user ? ` as @${esc(user)}` : ""} ${note}</summary>
          <div class="card"><div class="form-actions"><button type="button" class="btn btn-sm" id="gh-lock">Lock</button><button type="button" class="btn btn-sm btn-danger" id="gh-reset" style="margin-left:auto">Forget this device</button></div></div></details>`;
      else if (hasVault()) root.innerHTML = `<details class="more" open><summary>🔒 Private ${note}</summary>
          <div class="card"><div class="form-actions">${pw("gh-pw", "Password")}<button type="button" class="btn btn-primary" id="gh-open">Unlock</button><button type="button" class="btn btn-sm" id="gh-reset" style="margin-left:auto" title="Forget the saved token on this device">Reset</button></div></div></details>`;
      else root.innerHTML = `<details class="more" open><summary>🔒 Private · set up this device ${note}</summary>
          <div class="card"><div class="form-actions">${legacy ? "" : pw("gh-token", "GitHub token")}${pw("gh-new", "New password")}${pw("gh-new2", "Repeat password")}<button type="button" class="btn btn-primary" id="gh-setup">Save</button></div></div></details>`;
      const on = (id, fn) => root.querySelector("#" + id)?.addEventListener("click", fn);
      root.querySelectorAll("input").forEach(i => i.addEventListener("keydown", ev => { if (ev.key === "Enter") (root.querySelector("#gh-open") || root.querySelector("#gh-setup"))?.click(); }));
      on("gh-lock", () => { ssSet(SESSION_KEY, null); draw("locked"); onChange(); });
      on("gh-reset", () => { if (!confirm("Forget the saved token on this device? You will need the GitHub token again to set it up.")) return; ssSet(SESSION_KEY, null); lsSet(VAULT_KEY, null); lsSet(TOKEN_KEY, null); lsSet(USER_KEY, null); lsSet("tracker_cache", null); lsSet("leetcode_cache", null); draw(); onChange(); });
      on("gh-open", async () => {
        const p = root.querySelector("#gh-pw").value; if (!p) return;
        root.querySelector("#gh-open").disabled = true;
        const tok = await vaultOpen(p);
        if (!tok) { draw("wrong password"); return; }
        ssSet(SESSION_KEY, tok); draw("unlocked ✓"); onChange();
      });
      on("gh-setup", async () => {
        const tok = legacy ? clean(legacy) : clean(root.querySelector("#gh-token").value);
        const p1 = root.querySelector("#gh-new").value, p2 = root.querySelector("#gh-new2").value;
        if (!TOKEN_RE.test(tok)) { draw("that is not a GitHub token"); return; }
        if (p1.length < 6) { draw("password needs at least 6 characters"); return; }
        if (p1 !== p2) { draw("passwords differ"); return; }
        root.querySelector("#gh-setup").disabled = true; draw("checking…");
        ssSet(SESSION_KEY, tok);
        try {
          const u = await ghJson("/user");
          if (repo) {
            try { await ghJson(`/repos/${repo}`); } catch (e) { throw new Error(`cannot see ${repo} (${e.status})`); }
            try { await ghJson(`/repos/${repo}/commits?per_page=1`); } catch (e) { throw new Error(`no access to ${repo} (${e.message})`); }
          }
          await vaultSave(tok, p1); lsSet(TOKEN_KEY, null); lsSet(USER_KEY, u.login); draw("unlocked ✓");
        }
        catch (e) { ssSet(SESSION_KEY, null); draw(`rejected: ${e.message}`); return; }
        onChange();
      });
    };
    if (!window.isSecureContext || !window.crypto || !crypto.subtle) { root.innerHTML = `<div class="card small muted">🔒 Private · needs HTTPS</div>`; return; }
    draw();
  }
  window.GHVault = { token, hasVault, gh: ghJson, ghRaw, renderConnect, lsGet, lsSet };
})();
