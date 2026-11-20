"""Download the original checkpoint and reuse its verified browser cache entry."""

import io

import pyjs
import torch


WEIGHTS_URL = 'https://raw.githubusercontent.com/Anutosh21/dfine/1ef76758c3c4e5001a548d28dd763ca32d494ed6/content/dfine_s_coco.pth'
SHA256 = '48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89'
CACHE_NAME = 'dfine-small-weights-v1'

_FETCH = '''async (url, expectedHash, cacheName) => {
    let cache = null;
    let response;
    try {
        cache = await caches.open(cacheName);
        response = await cache.match(url);
    } catch { cache = null; }
    const cached = Boolean(response);
    if (!response) response = await fetch(url);
    if (!response.ok) throw Error(`Weights download failed: HTTP ${response.status}`);
    const data = await response.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)),
        byte => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== expectedHash) {
        if (cache) await cache.delete(url);
        throw Error('Pretrained weights failed the SHA-256 check; rerun to download again.');
    }
    let saved = cached;
    if (cache && !cached) {
        try {
            await cache.put(url, new Response(data));
            saved = true;
        } catch { /* Downloaded weights remain usable if storage is unavailable. */ }
    }
    return {data, cached, saved};
}'''


async def load_checkpoint():
    """Load the unchanged FP32 checkpoint, downloading only on a cache miss."""
    print('Loading pretrained weights (first download: 39.9 MiB)...', flush=True)
    result = await pyjs.js.eval(_FETCH)(WEIGHTS_URL, SHA256, CACHE_NAME)
    if pyjs.to_py(result.cached):
        print('Using verified cached weights.', flush=True)
    elif pyjs.to_py(result.saved):
        print('Weights verified and saved in the browser cache.', flush=True)
    else:
        print('Weights verified; browser storage is unavailable for caching.', flush=True)
    data = bytes(pyjs.to_py(result.data))
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
