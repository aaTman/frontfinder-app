// Local dev server for webapp/. Serves webapp/ at / and /data/* from either
// DATA_DIR (a local zarr output tree, with Range support) or, by default, a
// pass-through proxy to the production store at https://fronts.espr.ai/data.
//   node scripts/dev_server.mjs [port]            # proxy prod data
//   DATA_DIR=/path/to/output node scripts/dev_server.mjs 8080
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../webapp");
const port = Number(process.argv[2] || 8080);
const DATA_DIR = process.env.DATA_DIR ? path.resolve(process.env.DATA_DIR) : null;
const UPSTREAM = process.env.UPSTREAM || "https://fronts.espr.ai";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".svg": "image/svg+xml", ".geojson": "application/geo+json" };

function sendFile(req, res, file, extra = {}) {
  let st;
  try { st = fs.statSync(file); if (!st.isFile()) throw 0; } catch { res.writeHead(404); return res.end("not found"); }
  const headers = { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream", "Access-Control-Allow-Origin": "*", "Accept-Ranges": "bytes", ...extra };
  const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range || "");
  if (m) {
    let start = m[1] === "" ? st.size - Number(m[2]) : Number(m[1]);
    let end = m[1] !== "" && m[2] !== "" ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
    res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${st.size}`, "Content-Length": end - start + 1 });
    return fs.createReadStream(file, { start, end }).pipe(res);
  }
  res.writeHead(200, { ...headers, "Content-Length": st.size });
  fs.createReadStream(file).pipe(res);
}

http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const p = decodeURIComponent(url.pathname);
  if (p.startsWith("/data/")) {
    if (DATA_DIR) {
      const f = path.join(DATA_DIR, p.slice("/data/".length));
      if (!f.startsWith(DATA_DIR)) { res.writeHead(403); return res.end(); }
      const immutable = /\.zarr\//.test(p);
      return sendFile(req, res, f, { "Cache-Control": immutable ? "public, max-age=31536000, immutable" : "public, max-age=60" });
    }
    const headers = {};
    if (req.headers.range) headers.range = req.headers.range;
    const up = await fetch(UPSTREAM + p + url.search, { headers });
    const out = { "Access-Control-Allow-Origin": "*" };
    for (const h of ["content-type", "content-range", "accept-ranges", "cache-control", "etag"]) if (up.headers.get(h)) out[h] = up.headers.get(h);
    res.writeHead(up.status, out);
    return res.end(Buffer.from(await up.arrayBuffer()));
  }
  const rel = p === "/" ? "/index.html" : p;
  const f = path.join(root, rel);
  if (!f.startsWith(root)) { res.writeHead(403); return res.end(); }
  sendFile(req, res, fs.existsSync(f) ? f : path.join(root, "index.html"), { "Cache-Control": "no-store" });
}).listen(port, "127.0.0.1", () => console.log(`webapp on http://127.0.0.1:${port}  data: ${DATA_DIR ?? UPSTREAM + "/data (proxy)"}`));
