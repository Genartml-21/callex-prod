const http = require('http');
http.get('http://localhost:4000/api/simulation/agent-chat', (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => console.log('Status', res.statusCode, 'Data', data));
});
