// Usage: NODE_PATH=/tmp/pt/node_modules node scripts/profile_load.cjs http://127.0.0.1:8099/   (prints request waterfall: start ms, end ms, bytes, status, url)
// env WAIT=<ms> to extend observation (default 15000).
const puppeteer = require('puppeteer-core');
(async () => {
  const url = process.argv[2] || 'https://fronts.espr.ai/';
  const b = await puppeteer.launch({executablePath:'/home/orca/.local/bin/chromium', headless:'new', args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
  const p = await b.newPage(); await p.setViewport({width:1400,height:900});
  const cdp = await p.createCDPSession(); await cdp.send('Network.enable');
  if (process.argv[3]==='slow') await cdp.send('Network.emulateNetworkConditions',{offline:false,latency:80,downloadThroughput:1.5*1024*1024/8*8/ 8*1, uploadThroughput:750*1024/8}); 
  const t0 = Date.now(); const reqs = new Map();
  cdp.on('Network.requestWillBeSent', e => reqs.set(e.requestId,{url:e.request.url,start:Date.now()-t0,type:e.type,initiator:e.initiator.type}));
  cdp.on('Network.responseReceived', e => {const r=reqs.get(e.requestId); if(r){r.status=e.response.status; r.mime=e.response.mimeType; r.cache=e.response.fromDiskCache||e.response.fromServiceWorker?'cache':'';r.proto=e.response.protocol;}});
  cdp.on('Network.loadingFinished', e => {const r=reqs.get(e.requestId); if(r){r.end=Date.now()-t0; r.bytes=e.encodedDataLength;}});
  p.on('console', m => { const t=m.text(); if(/error|fail/i.test(t)) console.log('  [console]',t.slice(0,160)); });
  await p.goto(url,{waitUntil:'domcontentloaded'});
  await new Promise(r=>setTimeout(r, +process.env.WAIT||15000));
  const rows=[...reqs.values()].sort((a,b)=>a.start-b.start);
  let tot=0; for(const r of rows){tot+=r.bytes||0; console.log(String(r.start).padStart(5), String(r.end??'--').padStart(5), String(r.bytes??'').padStart(8), r.status, r.cache, r.url.replace(/^https:\/\//,'').slice(0,110));}
  console.log('requests',rows.length,'total bytes',tot);
  // Startup-to-first-data summary (same waterfall the lines above print, as
  // four numbers so runs can be diffed/medianed without eyeballing):
  //   first_chunk_start/end -- first `.zarr/<level>/<class>/c/` chunk request
  //   zstd_loaded           -- when the numcodecs zstd decoder finished
  //                            downloading (it used to be fetched only after
  //                            the first chunk response arrived)
  //   pre_requests/bytes    -- requests+bytes issued before that first chunk
  //                            request started
  const chunkRe=/\.zarr\/\d+\/[a-z_]+\/c\//i;
  const chunks=rows.filter(r=>chunkRe.test(r.url));
  const firstChunk=chunks[0];
  const zstd=rows.filter(r=>/numcodecs.*zstd/.test(r.url)&&r.end!==undefined);
  const pre=firstChunk?rows.filter(r=>r.start<firstChunk.start):[];
  console.log(['METRICS',
    `first_chunk_start=${firstChunk?firstChunk.start:'--'}`,
    `first_chunk_end=${chunks.length?Math.min(...chunks.map(r=>r.end??Infinity)): '--'}`,
    `zstd_loaded=${zstd.length?Math.max(...zstd.map(r=>r.end)):'never'}`,
    `pre_requests=${pre.length}`,
    `pre_bytes=${pre.reduce((s,r)=>s+(r.bytes||0),0)}`,
  ].join(' '));
  if (process.env.SHOT) { await p.screenshot({path:process.env.SHOT}); console.log('screenshot',process.env.SHOT); }
  await b.close();
})();
