// XRayNPC UI server.
// Serves public/ at "/", images/ at "/images", and /config.js which tells the
// page where the Python backend lives (env API_URL, default http://localhost:7860).
// PORT is read from the environment (Render sets it); locally it defaults to 3000
// so it never collides with the backend on 7860. .env is intentionally not loaded:
// its PORT belongs to the backend.
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = Number(process.env.PORT) || 3000;
const API_URL = (process.env.API_URL || 'http://localhost:7860').replace(/\/+$/, '');

app.disable('x-powered-by');

app.get('/healthz', (_req, res) => res.json({ ok: true }));

app.get('/config.js', (_req, res) => {
  const body = `window.XRAY_CONFIG = ${JSON.stringify({ apiBase: API_URL }).replace(/</g, '\\u003c')};`;
  res.type('application/javascript').set('Cache-Control', 'no-store').send(body);
});

// Editable site data: website_parameters.json and models/*.csv|json. Never cached, so edits show on refresh.
const noStore = res => res.set('Cache-Control', 'no-store');
app.get('/website_parameters.json', (_req, res) => {
  noStore(res);
  res.sendFile(path.join(__dirname, 'website_parameters.json'));
});
app.use(
  '/models',
  (req, res, next) => (/\.(csv|json)$/i.test(req.path) ? next() : res.sendStatus(404)),
  express.static(path.join(__dirname, 'models'), { index: false, setHeaders: noStore })
);
app.use('/models', (_req, res) => res.sendStatus(404));

app.use('/images', express.static(path.join(__dirname, 'images'), { maxAge: '1h' }));
app.use('/images', (_req, res) => res.sendStatus(404));

app.use(express.static(path.join(__dirname, 'public')));

app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`XRayNPC UI serving on http://localhost:${PORT} (API: ${API_URL})`);
});
