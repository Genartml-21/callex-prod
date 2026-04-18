const express = require('express');
const multer = require('multer');

const app = express();
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

app.patch('/test', upload.single('file'), (req, res) => {
    res.json({ body: req.body });
});

const server = app.listen(3002, async () => {
    console.log("Started test server");
    const fetch = (await import('node-fetch')).default;
    const res = await fetch('http://localhost:3002/test', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'active', name: 'Test Agent' })
    });
    const data = await res.json();
    console.log("JSON response:", data);
    server.close();
});
