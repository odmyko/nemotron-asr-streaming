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
}
async function harness(available=true){
  const elements={};
  const document={querySelector:selector=>elements[selector]??=new Element(),
    createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element('fragment'),
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
  const context=vm.createContext({document,WebSocket,URLSearchParams,ArrayBuffer,DataView,
    location:{protocol:'http:',host:'localhost:8080'},window:{AudioContext},
    navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop(){}}]})}},
    fetch:async url=>({json:async()=>url==='/health'?{diarization:{available}}:{languages:['auto','en-US']}})});
  const source=fs.readFileSync('web/index.html','utf8').split('<script>')[1].split('</script>')[0];
  vm.runInContext(source,context);
  await new Promise(setImmediate);
  return {elements,context,WebSocket,evaluate:expression=>vm.runInContext(expression,context)};
}
(async()=>{
  const ui=await harness();const e=ui.elements;
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
  ws.emit({type:'final',segments:[{text:'hello <script>',speaker_id:'speaker_0',final:true}]});
  assert.equal(e['#start'].disabled,false);
  assert.equal(e['#transcript'].textContent,'Speaker 1hello <script>');
  await e['#start'].onclick();
  const second=ui.WebSocket.last;second.emit({type:'ready'});
  second.emit({type:'partial',segments:[{text:'new recording',speaker_id:'speaker_0'}]});
  assert.equal(e['#transcript'].children.length,2);
  e['#clear'].onclick();
  assert.equal(e['#transcript'].textContent,'');
  second.emit({type:'partial',segments:[{text:'new recording continues',speaker_id:'speaker_0'},
                                      {text:'reply',speaker_id:'speaker_1'}]});
  assert.equal(e['#transcript'].textContent,'Speaker 1 continuesSpeaker 2reply');
  second.emit({type:'error',detail:'GPU error'});
  assert.equal(e['#start'].disabled,false);
  assert.match(e['#transcript'].textContent,/reply/);
  const unavailable=await harness(false);
  assert.equal(unavailable.elements['#diarize'].disabled,true);
  assert.equal(unavailable.elements['#start'].disabled,false);
  console.log('UI lifecycle, revised snapshots, safe rendering, Clear, errors, restart: OK');
})().catch(error=>{console.error(error);process.exitCode=1;});
