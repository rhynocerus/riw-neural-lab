"""Visualización educativa de la red XOR, con los mismos parámetros de entrenamiento."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([[0],[1],[1],[0]], dtype=float)
rng = np.random.default_rng(42)
W1 = rng.normal(size=(2,4)); b1 = np.zeros((1,4))
W2 = rng.normal(size=(4,1)); b2 = np.zeros((1,1))
def sig(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
errores=[]
for epoca in range(20000):
    h=sig(X@W1+b1); out=sig(h@W2+b2)
    errores.append(float(np.mean((out-Y)**2)))
    d2=(out-Y)*out*(1-out)
    d1=(d2@W2.T)*h*(1-h)
    W2-=h.T@d2/len(X); b2-=d2.mean(axis=0,keepdims=True)
    W1-=X.T@d1/len(X); b1-=d1.mean(axis=0,keepdims=True)
h=sig(X@W1+b1); pred=sig(h@W2+b2).flatten()
fig,ax=plt.subplots(1,2,figsize=(12,5))
fig.suptitle('RIW Neural LAB | Cinco neuronas, cuatro respuestas y ninguna magia',fontsize=13)
ax[0].plot(errores,linewidth=2)
ax[0].set(xlabel='Épocas',ylabel='Error cuadrático medio',title='La red aprende')
ax[0].grid(alpha=.25)
labels=['00','01','10','11']; bars=ax[1].bar(labels,pred)
ax[1].set(xlabel='Interruptores A y B',ylabel='Salida neuronal',title='Examen final',ylim=(0,1.08))
ax[1].axhline(.5,ls='--',color='gray',alpha=.6)
for bar,v in zip(bars,pred): ax[1].text(bar.get_x()+bar.get_width()/2,v+.025,f'{v:.3f}',ha='center')
fig.tight_layout()
outpath=Path(__file__).resolve().parent/'assets'/'aprendizaje_xor.png'
outpath.parent.mkdir(exist_ok=True)
fig.savefig(outpath,dpi=160,bbox_inches='tight')
print('Gráfico guardado en:',outpath)
plt.show()
