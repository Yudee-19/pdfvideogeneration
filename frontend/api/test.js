export default async function handler(req, res) {
  res.json({ 
    message: 'API function is working!',
    method: req.method,
    url: req.url,
    query: req.query
  });
}

