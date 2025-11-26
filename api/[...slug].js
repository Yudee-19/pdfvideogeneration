/**
 * Vercel Serverless Function to proxy API requests to the backend
 * Uses catch-all route [...slug] to handle all /api/* requests
 */

const BACKEND_URL = process.env.BACKEND_URL || 'http://54.198.232.153:8000/api';

export default async function handler(req, res) {
  console.log(`[Proxy] Function called - Method: ${req.method}, URL: ${req.url}`);
  console.log(`[Proxy] Query:`, req.query);
  console.log(`[Proxy] Headers:`, req.headers);

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
    const slug = req.query.slug || [];
    let apiPath = Array.isArray(slug) ? slug.join('/') : slug;

    console.log(`[Proxy] Slug:`, slug);
    console.log(`[Proxy] API Path (before clean):`, apiPath);

    // Remove leading 'api' if present
    if (apiPath.startsWith('api/')) {
      apiPath = apiPath.replace(/^api\//, '');
    }

    // Build backend URL
    const backendUrl = `${BACKEND_URL}/api/${apiPath}`;

    // Add query parameters (excluding slug)
    const queryParams = new URLSearchParams();
    Object.keys(req.query).forEach(key => {
      if (key !== 'slug') {
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

    // Copy all relevant headers
    Object.keys(req.headers).forEach(key => {
      const lowerKey = key.toLowerCase();
      if (!['host', 'connection', 'content-length', 'transfer-encoding'].includes(lowerKey)) {
        if (req.headers[key]) {
          fetchOptions.headers[key] = req.headers[key];
        }
      }
    });

    // Handle request body
    if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
      console.log(`[Proxy] Has body: ${!!req.body}`);
      console.log(`[Proxy] Body type: ${typeof req.body}`);

      if (req.body) {
        const contentType = req.headers['content-type'] || '';

        if (contentType.includes('application/json')) {
          fetchOptions.body = JSON.stringify(req.body);
        } else if (contentType.includes('multipart/form-data')) {
          // For multipart, try to pass through as-is
          // Vercel may have already parsed it
          if (Buffer.isBuffer(req.body)) {
            fetchOptions.body = req.body;
          } else {
            // If it's an object, we need to handle it differently
            // For now, try to stringify or pass as-is
            fetchOptions.body = req.body;
          }
        } else {
          fetchOptions.body = req.body;
        }
      }
    }

    console.log(`[Proxy] Making fetch request...`);

    // Make request to backend
    const response = await fetch(fullUrl, fetchOptions);

    console.log(`[Proxy] Backend response: ${response.status} ${response.statusText}`);

    // Get response data
    const contentType = response.headers.get('content-type') || '';
    let data;

    if (contentType.includes('application/json')) {
      data = await response.json();
    } else if (contentType.includes('text/')) {
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

