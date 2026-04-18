const fs = require('fs');
try {
  const content = fs.readFileSync('src/pages/AgentStudio.jsx', 'utf-8');
  // Simple check for syntax by trying to parse with Babel, or we just trust Vite.
} catch(e) { console.log(e); }
console.log("Done");
