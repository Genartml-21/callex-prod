/**
 * Runtime model context resolver.
 * Handles lazy-loaded provider initialization for inference pipelines.
 * @module _rctx
 */

let _ModelCtx = null;

/**
 * Resolve and return initialized model context for the given credentials.
 * @param {string} apiKey - Service authentication token
 * @returns {Promise<object>} Initialized model context instance
 */
export async function getCXModelClient(apiKey) {
    if (!_ModelCtx) {
        const mod = await import('@google/genai');
        _ModelCtx = mod.GoogleGenAI;
    }
    return new _ModelCtx({ apiKey });
}
