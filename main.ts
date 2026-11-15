// free-model-aggregator-01 entry: one server, two client wires, three upstream provider wires.
//
// Flow per trial: read the brief ($MAKEFUN_PROMPT_FILE), probe every provider's catalog route, build one
// catalog, listen on a free port, POST /register {protocol_version, port, catalog}, answer the client's
// requests (POST /v1/chat/completions as [OI], POST /v1/messages as Anthropic, GET /v1/models in the
// shape of its auth header), and exit 0 the moment GET /work says done.
//
// Routing: for every request, candidates are the free models that meet the manifest's requirements (a
// JSON-mode openai request is never routed to an anthropic-wire provider, whose wire cannot carry it).
// Candidates are tried strongest-rating first; a refusal (402, 429, 5xx) is retried twice on the same
// candidate and then blacklists that provider for the trial; 401/400 moves straight to the next
// candidate. Everything is a pure function of the brief and the providers' answers: no clock, no
// randomness, so a replay reproduces the same choices and the same catalog byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthReqToGemini,
  anthReqToOI,
  anthRespToOI,
  anthSseToOIEvents,
  anthropicError,
  anthropicModelList,
  gemRespToAnth,
  gemRespToOI,
  gemSseToAnthEvents,
  gemSseToOIEvents,
  oiReqToAnth,
  oiReqToGemini,
  oiRespToAnth,
  oiSseToAnthEvents,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Attempts per candidate before the provider is blacklisted for the trial. */
const ATTEMPTS_PER_CANDIDATE = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };

type ClientWire = 'openai' | 'anthropic';
type ProviderWire = 'openai' | 'anthropic' | 'gemini';

/** Live per-provider state for the failover policy. */
interface ProviderState {
  consecutiveRefusals: number;
  blacklisted: boolean;
}
const providerStates = new Map<string, ProviderState>();
const stateOf = (id: string): ProviderState => {
  let st = providerStates.get(id);
  if (st === undefined) {
    st = { consecutiveRefusals: 0, blacklisted: false };
    providerStates.set(id, st);
  }
  return st;
};

/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Whether the client asks for JSON mode on the openai wire (response_format json_object/json_schema). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * The candidates that may serve one request, strongest first: free models meeting the requirements,
 * never an anthropic-wire provider for a JSON-mode openai request (its wire cannot carry response_format).
 */
function candidatesFor(requires: Requirement, clientWire: ClientWire, jsonMode: boolean): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row: RatingRow | null = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, probe, model, rating: row !== null ? row.rating_milli : 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aSame = a.provider.wire === clientWire ? 0 : 1;
    const bSame = b.provider.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate: translate the client body into the provider's wire. */
function upstreamBodyFor(clientWire: ClientWire, providerWire: ProviderWire, body: Json, model: string, stream: boolean): Json {
  if (clientWire === 'openai' && providerWire === 'openai') {
    return { ...body, model, stream, ...(stream ? { stream_options: { include_usage: true } } : {}) };
  }
  if (clientWire === 'openai' && providerWire === 'anthropic') return oiReqToAnth(body, model);
  if (clientWire === 'openai' && providerWire === 'gemini') return oiReqToGemini(body, model);
  if (clientWire === 'anthropic' && providerWire === 'openai') return anthReqToOI(body, model);
  if (clientWire === 'anthropic' && providerWire === 'anthropic') return { ...body, model, stream };
  return anthReqToGemini(body, model);
}

/** POST one request to the candidate's provider route, in the provider's wire. */
async function callProvider(route: Route, clientWire: ClientWire, body: Json, stream: boolean): Promise<Response> {
  const wire = route.provider.wire;
  const base = joinBase(ENV_URL, route.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.provider.auth, route.provider.key),
  };
  let path: string;
  if (wire === 'openai') path = '/v1/chat/completions';
  else if (wire === 'anthropic') {
    path = '/v1/messages';
    headers['anthropic-version'] = '2023-06-01';
  } else {
    path = `/v1beta/models/${encodeURIComponent(route.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  }
  const upstream = upstreamBodyFor(clientWire, wire, body, route.model.id, stream);
  return fetch(`${base}${path}`, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

/** Relay the provider's stream bytes to the client as they arrive. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function streamHead(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/**
 * Try the candidates in order, strongest first; a refusal is retried twice on the same candidate, then
 * blacklists the provider for the trial. Answers through `answer`, which returns false when the upstream
 * answer could not be delivered (a shape we cannot relay), so the loop fails over.
 */
async function serveWithFailover(
  res: ServerResponse,
  clientWire: ClientWire,
  body: Json,
  stream: boolean,
  answer: (route: Route, upstream: Response) => Promise<boolean>,
): Promise<void> {
  const candidates = candidatesFor(requirementsFor(body), clientWire, jsonModeOf(body));
  for (const route of candidates) {
    const state = stateOf(route.provider.id);
    if (state.blacklisted) continue;
    let delivered = false;
    for (let attempt = 0; attempt < ATTEMPTS_PER_CANDIDATE && !delivered; attempt++) {
      let upstream: Response;
      try {
        upstream = await callProvider(route, clientWire, body, stream);
      } catch {
        state.consecutiveRefusals += 1;
        if (state.consecutiveRefusals >= ATTEMPTS_PER_CANDIDATE) state.blacklisted = true;
        continue;
      }
      if (upstream.status >= 200 && upstream.status < 300) {
        state.consecutiveRefusals = 0;
        delivered = await answer(route, upstream);
        if (!delivered) continue;
      } else {
        const refusal = upstream.status === 402 || upstream.status === 429 || upstream.status >= 500;
        try {
          await upstream.arrayBuffer();
        } catch {
          // The body is irrelevant on the failure path.
        }
        if (!refusal) break;
        state.consecutiveRefusals += 1;
        if (state.consecutiveRefusals >= ATTEMPTS_PER_CANDIDATE) state.blacklisted = true;
      }
    }
    if (delivered) return;
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'api_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

/** POST /v1/chat/completions: an [OI]-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  await serveWithFailover(res, 'openai', body, stream, async (route, upstream) => {
    const wire = route.provider.wire;
    if (wire === 'openai') {
      if (stream && upstream.body !== null) {
        streamHead(res);
        await pipeStream(upstream.body, res);
        return true;
      }
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(text);
      return true;
    }
    if (wire === 'anthropic') {
      if (stream && upstream.body !== null) {
        streamHead(res);
        res.end(await anthSseToOIEvents(upstream.body, route.model.id));
        return true;
      }
      const parsed: unknown = await upstream.json();
      sendJson(res, 200, anthRespToOI(isRecord(parsed) ? parsed : {}, route.model.id));
      return true;
    }
    if (stream && upstream.body !== null) {
      streamHead(res);
      res.end(await gemSseToOIEvents(upstream.body, route.model.id));
      return true;
    }
    const parsed: unknown = await upstream.json();
    sendJson(res, 200, gemRespToOI(isRecord(parsed) ? parsed : {}, route.model.id));
    return true;
  });
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  await serveWithFailover(res, 'anthropic', body, stream, async (route, upstream) => {
    const wire = route.provider.wire;
    if (wire === 'anthropic') {
      if (stream && upstream.body !== null) {
        streamHead(res);
        await pipeStream(upstream.body, res);
        return true;
      }
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(text);
      return true;
    }
    if (wire === 'openai') {
      if (stream && upstream.body !== null) {
        streamHead(res);
        res.end(await oiSseToAnthEvents(upstream.body, route.model.id));
        return true;
      }
      const parsed: unknown = await upstream.json();
      sendJson(res, 200, oiRespToAnth(isRecord(parsed) ? parsed : {}, route.model.id));
      return true;
    }
    if (stream && upstream.body !== null) {
      streamHead(res);
      res.end(await gemSseToAnthEvents(upstream.body, route.model.id));
      return true;
    }
    const parsed: unknown = await upstream.json();
    sendJson(res, 200, gemRespToAnth(isRecord(parsed) ? parsed : {}, route.model.id));
    return true;
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
