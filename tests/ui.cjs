// Exercise the actual inline UI script with deterministic DOM/audio/WS adapters.
// No real microphone, server, browser profile, or npm dependencies are needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this._text='';this.disabled=false;
    this.value='auto';this.checked=false;this.scrollHeight=500;this.scrollTop=450;this.clientHeight=50;
    this.style={setProperty:(name,value)=>{this[name]=value;}};}
  set textContent(value){this.children=[];this._text=String(value);}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  append(...children){this.children.push(...children);}
  appendChild(child){this.append(child);}
  replaceChildren(...children){this._text='';this.children=children.flatMap(c=>c.tag==='fragment'?c.children:[c]);}
  set innerHTML(_){throw new Error('Use text nodes for untrusted transcript content');}
  connect(){}
  disconnect(){}
  click(){this.onclick?.();}
  remove(){}
}
async function harness(available=true,failures=0){
  const elements={};
  const retries=[];
  const downloads=[],urls=new Map(),revoked=[];
  const document={querySelector:selector=>elements[selector]??=new Element(),
    body:new Element('body'),
    createElement:tag=>{const e=new Element(tag);
      if(tag==='a')e.onclick=()=>downloads.push({filename:e.download,blob:urls.get(e.href)});
      return e;},createDocumentFragment:()=>new Element('fragment'),
    createTextNode:text=>({textContent:text})};
  class AudioContext {
    constructor(){this.sampleRate=48000;this.state='running';}
    async resume(){}
    async close(){this.state='closed';}
    createMediaStreamSource(){return new Element();}
    createScriptProcessor(){return new Element();}
  }
  class WebSocket {
    static OPEN=1;
    constructor(url){this.url=url;this.readyState=1;this.bufferedAmount=0;this.sent=[];WebSocket.last=this;}
    send(data){this.sent.push(data);}
    close(){this.readyState=3;this.onclose?.();}
    emit(message){this.onmessage({data:JSON.stringify(message)});}
  }
  const context=vm.createContext({document,WebSocket,URLSearchParams,ArrayBuffer,DataView,Blob,
    URL:{createObjectURL:blob=>{const id=`blob:${urls.size}`;urls.set(id,blob);return id;},
      revokeObjectURL:url=>revoked.push(url)},
    FormData:class{append(){}},
    location:{protocol:'http:',host:'localhost:8080'},window:{AudioContext},
    navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop(){}}]})}},
    setTimeout:callback=>retries.push(callback),
    fetch:async url=>{
      if(url==='/health'&&failures-->0)throw new Error('Server starting');
      if(url==='/transcribe')return {ok:true,json:async()=>({text:'File result',transcript_id:'file-1',backup:{status:'saved',filename:'file-1.json'}})};
      return {ok:true,json:async()=>url==='/health'?{status:'ok',diarization:{available}}:{languages:['auto','en-US']}};
    }});
  const source=fs.readFileSync('web/index.html','utf8').split('<script>')[1].split('</script>')[0];
  vm.runInContext(source,context);
  await new Promise(setImmediate);
  return {elements,context,WebSocket,retries,downloads,revoked,evaluate:expression=>vm.runInContext(expression,context)};
}
(async()=>{
  const ui=await harness();const e=ui.elements;
  assert.equal(e['#exportJson'].disabled,true);
  e['#diarize'].checked=true;
  await e['#start'].onclick();
  const ws=ui.WebSocket.last;
  assert.match(ws.url,/diarize=true/);
  ws.emit({type:'ready'});
  assert.equal(e['#start'].disabled,true);
  assert.equal(e['#diarize'].disabled,true);
  ws.emit({type:'partial',segments:[{text:'hel',speaker_id:'speaker_0',final:false}]});
  ws.emit({type:'partial',segments:[{text:'hello <script>',speaker_id:'speaker_0',final:false}]});
  assert.equal(e['#transcript'].textContent,'Speaker 1hello <script>');
  assert.equal(e['#transcript'].children.length,1);
  e['#stop'].onclick();
  assert.match(e['#statusText'].textContent,/^finishing/);
  assert.equal(e['#start'].disabled,true);
  assert.equal(JSON.parse(ws.sent.at(-1)).type,'end');
  ws.emit({type:'final',transcript_id:'saved-first',backup:{status:'saved',filename:'first.json'},
    segments:[{text:'hello <script>',speaker_id:'speaker_0',start:0,end:1.25,final:true}]});
  assert.equal(e['#start'].disabled,false);
  assert.equal(e['#transcript'].textContent,'Speaker 1hello <script>');
  assert.match(e['#backupStatus'].textContent,/Saved on server: first.json/);
  e['#exportTxt'].onclick();
  assert.match(ui.downloads.at(-1).filename,/\.txt$/);
  assert.match(await ui.downloads.at(-1).blob.text(),/Speaker 1 \[0.00–1.25s\]: hello <script>/);
  e['#exportMd'].onclick();
  assert.match(await ui.downloads.at(-1).blob.text(),/\\<script\\>/);
  e['#exportJson'].onclick();
  const exported=JSON.parse(await ui.downloads.at(-1).blob.text());
  assert.equal(exported.segments[0].recording_id,'saved-first');
  assert.equal(exported.segments[0].end,1.25);
  assert.equal(exported.segments[0].final,true);
  await e['#start'].onclick();
  const second=ui.WebSocket.last;second.emit({type:'ready'});
  second.emit({type:'partial',segments:[{text:'new recording',speaker_id:'speaker_0'}]});
  assert.equal(e['#transcript'].children.length,2);
  e['#exportMd'].onclick();
  const combined=await ui.downloads.at(-1).blob.text();
  assert.match(combined,/## Recording 1/);assert.match(combined,/## Recording 2/);
  e['#clear'].onclick();
  assert.equal(e['#transcript'].textContent,'');
  assert.equal(e['#exportTxt'].disabled,true);
  second.emit({type:'partial',segments:[{text:'new recording continues',speaker_id:'speaker_0'},
                                      {text:'reply',speaker_id:'speaker_1'}]});
  assert.equal(e['#transcript'].textContent,'Speaker 1 continuesSpeaker 2reply');
  e['#exportJson'].onclick();
  const afterClear=JSON.parse(await ui.downloads.at(-1).blob.text());
  assert.equal(afterClear.segments.length,2);
  assert.equal(afterClear.segments[0].text,' continues');
  assert.equal(afterClear.segments[0].final,false);
  second.emit({type:'error',detail:'GPU error'});
  assert.equal(e['#start'].disabled,false);
  assert.match(e['#transcript'].textContent,/reply/);
  assert.match(e['#backupStatus'].textContent,/not confirmed/);
  const failedBackup=await harness();
  await failedBackup.elements['#start'].onclick();
  failedBackup.WebSocket.last.emit({type:'ready'});
  failedBackup.WebSocket.last.emit({type:'final',text:'Привіт 🌍',backup:{status:'error'}});
  assert.match(failedBackup.elements['#backupStatus'].textContent,/backup failed/);
  failedBackup.elements['#exportTxt'].onclick();
  assert.equal(await failedBackup.downloads[0].blob.text(),'Привіт 🌍\n');
  failedBackup.retries.pop()();assert.equal(failedBackup.revoked.length,1);
  const upload=await harness();
  upload.elements['#fileInput'].files=[{name:'demo.wav'}];
  await upload.elements['#upload'].onclick();
  upload.elements['#exportJson'].onclick();
  const fileExport=JSON.parse(await upload.downloads[0].blob.text());
  assert.equal(fileExport.segments[0].source,'file');
  assert.equal(fileExport.segments[0].filename,'demo.wav');
  const unavailable=await harness(false);
  assert.equal(unavailable.elements['#diarize'].disabled,true);
  assert.equal(unavailable.elements['#start'].disabled,false);
  const starting=await harness(true,1);
  assert.equal(starting.elements['#start'].disabled,true);
  assert.match(starting.elements['#statusText'].textContent,/waiting for server/);
  await starting.retries.shift()();
  assert.equal(starting.elements['#start'].disabled,false);
  assert.equal(starting.elements['#diarize'].disabled,false);
  console.log('UI lifecycle, exports (TXT/MD/JSON), backups, Clear, uploads, errors, restart: OK');
})().catch(error=>{console.error(error);process.exitCode=1;});
