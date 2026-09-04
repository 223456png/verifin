#!/usr/bin/env node
/**
 * VeriFin LLM 桥接服务 v2：429 感知重试 + 最小请求间隔。
 *
 * 实测（2026-09-04 L1 run）：z-ai 后端对持续调用返回 429
 * "Too many requests"——v1 桥接把它当一次性错误直接 500，
 * Python 侧 2 次短退避（1s/2s）不足以越过限流窗口，导致
 * planner 240/260 调用失败降级。
 *
 * v2 策略：
 * - 全局限流：请求间最小间隔 MIN_GAP_MS（默认 4000ms）
 * - 429/5xx 桥接内重试：指数退避 5s→10s→20s→40s→60s（最多 5 次）
 * - 重试期间串行队列自然阻塞后续请求（benchmark 本就串行）
 */

import ZAI from 'z-ai-web-dev-sdk';
import http from 'node:http';

const PORT = Number(process.env.BRIDGE_PORT || 8642);
const MIN_GAP_MS = Number(process.env.MIN_GAP_MS || 4000);
const RETRY_DELAYS = [5000, 10000, 20000, 40000, 60000];

let zai = null;
let queue = Promise.resolve();
let lastRequestAt = 0;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function init() {
  zai = await ZAI.create();
  console.log(`[bridge] Z-AI SDK 初始化完成（min_gap=${MIN_GAP_MS}ms）`);
}

async function callWithRetry(messages) {
  let lastErr = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS.length; attempt++) {
    // 全局最小间隔：距上次请求不足 MIN_GAP 则等待
    const wait = lastRequestAt + MIN_GAP_MS - Date.now();
    if (wait > 0) await sleep(wait);
    lastRequestAt = Date.now();
    try {
      const completion = await zai.chat.completions.create({
        messages,
        thinking: { type: 'disabled' },
      });
      const text = completion.choices?.[0]?.message?.content;
      if (typeof text !== 'string' || !text.trim()) {
        throw new Error('empty response');
      }
      return text;
    } catch (err) {
      lastErr = err;
      const msg = String(err?.message || err);
      const retryable =
        msg.includes('429') ||
        msg.includes('500') ||
        msg.includes('502') ||
        msg.includes('503') ||
        msg.includes('empty response');
      if (!retryable || attempt === RETRY_DELAYS.length) {
        throw err;
      }
      const delay = RETRY_DELAYS[attempt];
      console.warn(`[bridge] 第${attempt + 1}次失败（${msg.slice(0, 80)}），${delay}ms 后重试`);
      await sleep(delay);
    }
  }
  throw lastErr;
}

function handleComplete(body) {
  const prompt = String(body.prompt ?? '');
  const system = body.system ? String(body.system) : null;
  if (!prompt.trim()) {
    return Promise.resolve({ status: 400, payload: { error: 'empty prompt' } });
  }
  const messages = system
    ? [{ role: 'assistant', content: system }, { role: 'user', content: prompt }]
    : [{ role: 'user', content: prompt }];
  const task = queue.then(async () => {
    try {
      const text = await callWithRetry(messages);
      return { status: 200, payload: { text } };
    } catch (err) {
      console.error('[bridge] 最终失败:', String(err?.message || err).slice(0, 120));
      return { status: 500, payload: { error: String(err?.message || err) } };
    }
  });
  // 队列吞掉异常，保证后续请求不受影响
  queue = task.catch(() => {});
  return task;
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }
  if (req.method === 'POST' && req.url === '/complete') {
    let raw = '';
    req.on('data', (c) => (raw += c));
    req.on('end', () => {
      let body;
      try {
        body = JSON.parse(raw || '{}');
      } catch {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'bad json' }));
        return;
      }
      handleComplete(body).then(({ status, payload }) => {
        res.writeHead(status, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(payload));
      });
    });
    return;
  }
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

server.requestTimeout = 300_000;
server.headersTimeout = 310_000;

init()
  .then(() => {
    server.listen(PORT, '127.0.0.1', () => {
      console.log(`[bridge] listening on http://127.0.0.1:${PORT}`);
    });
  })
  .catch((err) => {
    console.error('[bridge] SDK 初始化失败:', err?.message || err);
    process.exit(1);
  });
