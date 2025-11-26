/**
 * Vercel Serverless Function to proxy API requests to the backend
 * Uses catch-all route [...slug] to handle all /api/* requests
 */

const BACKEND_URL = "/api";
const busboy = require('busboy');

export default async function handler(req, res) {
  // Log immediately to confirm function is being called
  console.log(`[Proxy] ========== FUNCTION CALLED ==========`);
  console.log(`[Proxy] Method: ${req.method}`);
  console.log(`[Proxy] URL: ${req.url}`);
  console.log(`[Proxy] Pathname: ${req.url?.split('?')[0]}`);
  console.log(`[Proxy] Query:`, JSON.stringify(req.query));
  console.log(`[Proxy] BACKEND_URL: ${process.env.BACKEND_URL || 'NOT SET - using default'}`);

  // Set CORS headers first
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, PATCH');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle OPTIONS preflight
  if (req.method === 'OPTIONS') {
    console.log('[Proxy] Handling OPTIONS request');
    return res.status(200).end();
  }

  // Log all methods
  console.log(`[Proxy] Handling ${req.method} request`);

  try {
    // Get the path segments from the slug parameter
    // Vercel uses '...slug' as the query key for catch-all routes
    const slugParam = req.query['...slug'] || req.query.slug;
    let slug = slugParam || [];

    // Handle both array and string formats
    if (typeof slug === 'string') {
      slug = slug.split('/').filter(Boolean);
    } else if (!Array.isArray(slug)) {
      slug = [];
    }

    // Alternative: Parse from req.url if slug is empty (fallback for edge cases)
    if (slug.length === 0 && req.url) {
      const urlPath = req.url.split('?')[0]; // Remove query string
      // Remove /api prefix if present
      const pathWithoutApi = urlPath.startsWith('/api/') ? urlPath.slice(5) : urlPath.slice(1);
      if (pathWithoutApi) {
        slug = pathWithoutApi.split('/').filter(Boolean);
        console.log(`[Proxy] Fallback: Parsed path from req.url:`, pathWithoutApi);
      }
    }

    let apiPath = slug.join('/');

    console.log(`[Proxy] Slug param:`, slugParam);
    console.log(`[Proxy] Slug array:`, slug);
    console.log(`[Proxy] API Path (before clean):`, apiPath);

    // Remove leading 'api' if present (shouldn't happen, but just in case)
    if (apiPath.startsWith('api/')) {
      apiPath = apiPath.replace(/^api\//, '');
    }

    // Build backend URL
    // Most endpoints are under /api/, but /health is at root level
    // Root-level endpoints: '', 'health'
    const rootEndpoints = ['', 'health'];
    const isRootEndpoint = rootEndpoints.includes(apiPath);

    // Construct the backend URL
    // Encode each path segment separately to handle spaces and special characters
    let backendUrl;
    if (isRootEndpoint) {
      // Root-level endpoints: / or /health
      backendUrl = apiPath === '' ? `${BACKEND_URL}/` : `${BACKEND_URL}/${apiPath}`;
    } else {
      // All other endpoints are under /api/
      // Split the path and encode each segment separately
      const pathSegments = apiPath.split('/').map(segment => encodeURIComponent(segment));
      const encodedPath = pathSegments.join('/');
      backendUrl = `${BACKEND_URL}/api/${encodedPath}`;
    }

    // Add query parameters (excluding slug-related keys)
    const queryParams = new URLSearchParams();
    Object.keys(req.query).forEach(key => {
      if (key !== 'slug' && key !== '...slug') {
        queryParams.append(key, req.query[key]);
      }
    });

    const queryString = queryParams.toString();
    const fullUrl = queryString ? `${backendUrl}?${queryString}` : backendUrl;

    console.log(`[Proxy] Backend URL: ${fullUrl}`);
    console.log(`[Proxy] Content-Type: ${req.headers['content-type'] || 'none'}`);

    // Prepare fetch options
    const fetchOptions = {
      method: req.method,
      headers: {},
    };

    // Copy only essential headers (exclude Vercel-specific and problematic headers)
    const excludedHeaders = [
      'host', 'connection', 'content-length', 'transfer-encoding',
      'x-vercel-', 'x-forwarded-', 'forwarded', 'sec-fetch-', 'sec-ch-ua',
      'accept-encoding', 'accept-language', 'referer', 'origin', 'priority'
    ];

    Object.keys(req.headers).forEach(key => {
      const lowerKey = key.toLowerCase();
      const shouldExclude = excludedHeaders.some(excluded => lowerKey.startsWith(excluded));
      if (!shouldExclude && req.headers[key]) {
        fetchOptions.headers[key] = req.headers[key];
      }
    });

    // Helper function to handle backend response
    const handleResponse = async (response) => {
      console.log(`[Proxy] Backend response: ${response.status} ${response.statusText}`);

      // Get response data
      const responseContentType = response.headers.get('content-type') || '';
      let data;

      if (responseContentType.includes('application/json')) {
        data = await response.json();
      } else if (responseContentType.includes('text/')) {
        data = await response.text();
      } else {
        data = await response.arrayBuffer();
      }

      // Set response status and headers
      res.status(response.status);

      // Copy response headers
      response.headers.forEach((value, key) => {
        const lowerKey = key.toLowerCase();
        if (!['connection', 'transfer-encoding', 'content-encoding'].includes(lowerKey)) {
          res.setHeader(key, value);
        }
      });

      // Send response
      if (data instanceof ArrayBuffer) {
        res.send(Buffer.from(data));
      } else {
        res.json(data);
      }

      console.log(`[Proxy] Response sent successfully`);
    };

    // Handle request body
    if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
      const contentType = req.headers['content-type'] || '';
      console.log(`[Proxy] Has body: ${!!req.body}`);
      console.log(`[Proxy] Body type: ${typeof req.body}`);
      console.log(`[Proxy] Content-Type: ${contentType}`);

      if (contentType.includes('multipart/form-data')) {
        // For multipart/form-data, forward the raw body directly
        // This avoids parsing/reconstruction issues and preserves the original format
        console.log('[Proxy] Handling multipart/form-data - forwarding raw body');

        return new Promise((resolve, reject) => {
          // Buffer the request body
          const chunks = [];
          req.on('data', (chunk) => {
            chunks.push(chunk);
          });

          req.on('end', () => {
            const bodyBuffer = Buffer.concat(chunks);
            console.log(`[Proxy] Request body buffered: ${bodyBuffer.length} bytes`);

            // Forward the raw multipart body directly to backend
            // Preserve the original Content-Type header with boundary
            fetchOptions.body = bodyBuffer;
            fetchOptions.headers['Content-Type'] = contentType; // Preserve original boundary

            // Remove content-length - let fetch calculate it from the buffer
            delete fetchOptions.headers['content-length'];

            console.log('[Proxy] Forwarding raw multipart body to backend...');
            console.log('[Proxy] Content-Type:', fetchOptions.headers['Content-Type']);
            console.log('[Proxy] Body size:', bodyBuffer.length, 'bytes');

            // Make request to backend
            fetch(fullUrl, fetchOptions)
              .then(async (response) => {
                console.log(`[Proxy] Backend response: ${response.status} ${response.statusText}`);
                const responseText = await response.text();
                console.log('[Proxy] Backend response body:', responseText.substring(0, 500));

                // Set response status and headers
                res.status(response.status);

                // Copy response headers
                response.headers.forEach((value, key) => {
                  const lowerKey = key.toLowerCase();
                  if (!['connection', 'transfer-encoding', 'content-encoding'].includes(lowerKey)) {
                    res.setHeader(key, value);
                  }
                });

                // Try to parse as JSON, fallback to text
                try {
                  const data = JSON.parse(responseText);
                  res.json(data);
                } catch {
                  res.send(responseText);
                }
                resolve();
              })
              .catch(error => {
                console.error('[Proxy] Backend request error:', error);
                console.error('[Proxy] Error stack:', error.stack);
                res.status(500).json({
                  error: 'Proxy error',
                  message: error.message
                });
                resolve();
              });
          });

          req.on('error', (error) => {
            console.error('[Proxy] Request stream error:', error);
            res.status(400).json({
              error: 'Failed to read request body',
              message: error.message
            });
            resolve();
          });
        });
      } else if (contentType.includes('application/json')) {
        fetchOptions.body = JSON.stringify(req.body);
        fetchOptions.headers['Content-Type'] = 'application/json';
      } else if (req.body) {
        fetchOptions.body = req.body;
      }
    }

    // For multipart, the Promise above handles the request and returns early
    // For other content types, continue with normal flow
    if (!req.headers['content-type']?.includes('multipart/form-data')) {
      console.log(`[Proxy] Making fetch request...`);

      // Make request to backend
      const response = await fetch(fullUrl, fetchOptions);
      await handleResponse(response);
    }

  } catch (error) {
    console.error('[Proxy] Error:', error.message);
    console.error('[Proxy] Stack:', error.stack);
    console.error('[Proxy] Full error:', error);

    res.status(500).json({
      error: 'Proxy error',
      message: error.message,
      backendUrl: BACKEND_URL,
      requestMethod: req.method,
      requestUrl: req.url
    });
  }
}
