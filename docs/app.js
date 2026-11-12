"use strict";
// RIW Neural Experience: red XOR real 2→4→1 con retropropagación.
const $=id=>document.getElementById(id);
const X=[[0,0],[0,1],[1,0],[1,1]], Y=[0,1,1,0];
const sig=x=>1/(1+Math.exp(-Math.max(-40,Math.min(40,x))));
let w1,b1,w2,b2,epoch=0,history=[];
function initialize(){
  w1=Array.from({length:2},()=>Array.from({length:4},()=>Math.random()*2-1));
  b1=Array(4).fill(0); w2=Array.from({length:4},()=>Math.random()*2-1);b2=0;epoch=0;history=[];
  update();
}
function predict(x){const h=Array.from({length:4},(_,j)=>sig(x[0]*w1[0][j]+x[1]*w1[1][j]+b1[j]));return {h,out:sig(h.reduce((v,hj,j)=>v+hj*w2[j],b2))};}
function loss(){return X.reduce((v,x,i)=>v+(predict(x).out-Y[i])**2,0)/4;}
function trainBatch(){const gw1=Array.from({length:2},()=>Array(4).fill(0)),gb1=Array(4).fill(0),gw2=Array(4).fill(0);let gb2=0;
  for(let i=0;i<4;i++){
    const {h,out}=predict(X[i]);const d2=(out-Y[i])*out*(1-out);
    for(let j=0;j<4;j++){
      const d1=d2*w2[j]*h[j]*(1-h[j]);
      for(let k=0;k<2;k++)gw1[k][j]+=X[i][k]*d1;
      gb1[j]+=d1; gw2[j]+=h[j]*d2;
    }gb2+=d2;
  }
  for(let j=0;j<4;j++){
    for(let k=0;k<2;k++)w1[k][j]-=gw1[k][j]/4;
    b1[j]-=gb1[j]/4;w2[j]-=gw2[j]/4;
  }b2-=gb2/4;epoch++;
}
function network(){const svg=$('net');if(!svg)return;let s='';const a=[65,195],h=[35,99,163,227],end=130;
  function line(x1,y1,x2,y2,w){const color=w>=0?'#5be0ce':'#efbe6c',sw=Math.max(1,Math.min(8,Math.abs(w)*2.8));return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}" opacity=".55"/>`;}
  for(let i=0;i<2;i++)for(let j=0;j<4;j++)s+=line(52,a[i],207,h[j],w1[i][j]);
  for(let j=0;j<4;j++)s+=line(207,h[j],363,end,w2[j]);
  for(const y of a)s+=`<circle cx="52" cy="${y}" r="14" fill="#2b6c83" stroke="#8fe7e7" stroke-width="2"/>`;
  for(const y of h)s+=`<circle cx="207" cy="${y}" r="14" fill="#375e83" stroke="#accdf7" stroke-width="2"/>`;
  s+=`<circle cx="363" cy="${end}" r="17" fill="#a16c33" stroke="#ffd48e" stroke-width="2"/>`;
  s+='<g fill="#cbdde9" font-size="12" text-anchor="middle"><text x="52" y="254">Entradas</text><text x="207" y="254">Capa oculta</text><text x="363" y="254">Salida</text></g>';
  svg.innerHTML=s;
}
function plot(){const c=$('plot');const rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);c.width=Math.max(400,Math.round(rect.width*dpr));c.height=Math.round(250*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const W=c.width/dpr,H=250;ctx.clearRect(0,0,W,H);
  const left=45,right=W-20,top=15,bottom=H-28;ctx.strokeStyle='#5a7188';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
  ctx.fillStyle='#c2d1e0';ctx.font='12px sans-serif';ctx.fillText('Error',6,22);ctx.fillText('Épocas',right-45,H-5);
  const max=Math.max(.30,...history.map(x=>x[1]));
  ctx.strokeStyle='#65e0ce';ctx.lineWidth=2;ctx.beginPath();
  history.forEach(([ep,v],i)=>{const x=left+(right-left)*(ep/Math.max(1,epoch)),y=bottom-(bottom-top)*v/max;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
  ctx.fillText(max.toFixed(2),5,top+10);ctx.fillText('0',30,bottom+15);ctx.fillText(String(epoch),right-35,bottom+15);
}
function update(){const a=Number($('switchA').checked),b=Number($('switchB').checked),expected=a^b;const p=predict([a,b]).out;const classification=Number(p>=.5);$('epoch').textContent=epoch.toLocaleString('es-ES');$('loss').textContent=loss().toFixed(5);$('expected').textContent=`Regla correcta: ${expected?'encendida':'apagada'} (${a} XOR ${b} = ${expected}).`;$('prediction').textContent=(classification?'Encendida':'Apagada')+` · ${(p*100).toFixed(1)}% de activación`;$('confidence').textContent='La activación no es una probabilidad calibrada';$('verdict').textContent=epoch===0?'Red sin entrenar: sus predicciones pueden fallar.':(X.every((x,i)=>Number(predict(x).out>=.5)===Y[i])?'La red acierta las 4 combinaciones XOR.':'Todavía se equivoca en alguna combinación; puedes seguir entrenando.');$('bulb').classList.toggle('on',!!expected);network();plot();}
$('switchA').addEventListener('change',update);$('switchB').addEventListener('change',update);$('reset').addEventListener('click',initialize);
$('train').addEventListener('click',()=>{const button=$('train');button.disabled=true;button.textContent='Entrenando…';setTimeout(()=>{history.push([epoch,loss()]);for(let i=0;i<2000;i++){trainBatch();if(i%20===19)history.push([epoch,loss()]);}update();button.disabled=false;button.textContent='Entrenar 2.000 épocas';},20);});
window.addEventListener('resize',plot);initialize();
