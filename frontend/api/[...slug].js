/**
 * Vercel Serverless Function to proxy API requests to the backend
 * Uses catch-all route [...slug] to handle all /api/* requests
 */

const BACKEND_URL = process.env.BACKEND_URL || 'http://54.198.232.153:8000';
const busboy = require('busboy');

export default async function handler(req, res) {
  console.log(`[Proxy] ========== FUNCTION CALLED ==========`);
  console.log(`[Proxy] Method: ${req.method}`);
  console.log(`[Proxy] URL: ${req.url}`);
  console.log(`[Proxy] Query:`, JSON.stringify(req.query));
  console.log(`[Proxy] BACKEND_URL: ${process.env.BACKEND_URL || 'NOT SET - using default'}`);
  console.log(`[Proxy] Headers:`, JSON.stringify(req.headers));
  
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
    let backendUrl;
    if (isRootEndpoint) {
      // Root-level endpoints: / or /health
      backendUrl = apiPath === '' ? `${BACKEND_URL}/` : `${BACKEND_URL}/${apiPath}`;
    } else {
      // All other endpoints are under /api/
      backendUrl = `${BACKEND_URL}/api/${apiPath}`;
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
    
    // Copy all relevant headers
    Object.keys(req.headers).forEach(key => {
      const lowerKey = key.toLowerCase();
      if (!['host', 'connection', 'content-length', 'transfer-encoding'].includes(lowerKey)) {
        if (req.headers[key]) {
          fetchOptions.headers[key] = req.headers[key];
        }
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
        // Vercel serverless functions don't parse multipart automatically
        // We need to parse it using busboy and reconstruct it for the backend
        console.log('[Proxy] Handling multipart/form-data');
        console.log('[Proxy] Body is buffer:', Buffer.isBuffer(req.body));
        console.log('[Proxy] Body type:', typeof req.body);
        
        // Parse multipart data using busboy
        return new Promise((resolve, reject) => {
          const FormData = require('form-data');
          const formData = new FormData();
          const bb = busboy({ headers: req.headers });
          
          // Collect all fields and files
          const fields = {};
          const files = [];
          
          let fileCount = 0;
          let fieldCount = 0;
          let completedFiles = 0;
          let sent = false; // Guard to prevent multiple sends
          
          const sendToBackend = () => {
            if (sent) {
              console.log('[Proxy] sendToBackend already called, skipping');
              return;
            }
            sent = true;
            console.log(`[Proxy] All data collected. Files: ${files.length}, Fields: ${Object.keys(fields).length}`);
            
            // Reconstruct FormData for backend
            for (const [key, value] of Object.entries(fields)) {
              formData.append(key, value);
            }
            
            for (const file of files) {
              console.log(`[Proxy] Appending file ${file.name}: ${file.buffer.length} bytes, filename: ${file.filename}`);
              formData.append(file.name, file.buffer, {
                filename: file.filename,
                contentType: file.mimeType || 'application/octet-stream'
              });
            }
            
            // Update headers
            delete fetchOptions.headers['content-type'];
            fetchOptions.body = formData;
            const formHeaders = formData.getHeaders();
            console.log('[Proxy] FormData headers:', formHeaders);
            Object.assign(fetchOptions.headers, formHeaders);
            
            console.log('[Proxy] FormData reconstructed, making backend request...');
            console.log('[Proxy] Request headers:', fetchOptions.headers);
            
            // Make request to backend
            fetch(fullUrl, fetchOptions)
              .then(async (response) => {
                console.log(`[Proxy] Backend response: ${response.status} ${response.statusText}`);
                const responseText = await response.text();
                console.log('[Proxy] Backend response body:', responseText.substring(0, 500)); // Log first 500 chars
                
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
          };
          
          bb.on('finish', () => {
            console.log('[Proxy] Busboy finished parsing');
            // sendToBackend will be called when all files and fields are collected
            // But also check here in case everything was already collected
            if (completedFiles === fileCount && Object.keys(fields).length === fieldCount) {
              sendToBackend();
            }
          });
          
          bb.on('error', (error) => {
            console.error('[Proxy] Busboy error:', error);
            res.status(400).json({
              error: 'Failed to parse multipart data',
              message: error.message
            });
            resolve();
          });
          
          // Pipe the request to busboy
          req.pipe(bb);
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
