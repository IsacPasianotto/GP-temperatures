- slurm-417955
    - 1 master, 1 slave   (scale 128)
    - 1k points
    - batch size: 100
    - iter train: 100
    - init time: 9.33
    - train time: 63.666
- slurm-417971
    - 1 master, 2 slave (scale 256)
    - 1k points
    - batch size: 100
    - iter train: 100
    - init time: 6.32 
    - train time: 23.96


Quindi scala bene rispetto al numero di nodi


- slurm-417975
    - 1 master, 2 slave (scale 256)
    - 2 k points
    - batch size: 100
    - iter train: 100
    - init time: 21.97
    - train time: 443.20

(Sembra brutto, ma sta scalando quadraticamente anzichè cubicamente, che è bello :) )

Provando a variare la batch-size:

- slurm-417982
    - 1 master, 2 slave (scale 256)
    - 2k points
    - batchsize = 15626   (2k\*2k da calcolare) / (128cores \* 2 nodi)
    - iter train: 100
    - init time: 119 sec 
    - train time: lo ho interrotto a mano dopo 10 min 

Pessima idea, proviamo ad aumentarla, ma meno, es 2ki/2\*n nodi

- slurm-417989
    - 1 master, 2 slave (scale 256)
    - 2k points
    - batchsize = 500
    - iter train: 100
    - init time: 17.43  
    - train time: circa 400 (crashato perchè ho modificato il codice e una chiamata alla libreria è fallita post train) 


batch troppo grosse appesantiscono i nodi e basta, non c'è un numero relazionale rispetto alle risorse usate o cose del genere, è un parametro più hardware dependent che altro.
Tra l'altro con alcuni numeri sfortunati la libreria crasha.
Notare dalla run successiva che mettendo un buon valore di batch size i tempi si riducono (init di 25 point comparabile con quello di 2k point)

PROVIAMO UNA RUN GROSSA: 

- slurm-418026
    - 1 master, 2 slave (scale 256)
    - 25k points
    - batchsize: 1000
    - iter train: 100
    - init time: 105.34
    - train time: 5159.07 

- slurm-418473
    731 master, 3 slaves
    - 45k points
    - batch size: 750
    - iter train: 100
    - init time: 5-600 sec 
    - train time: CRASH di CHEPH??? Sometihing weired


- - - 

a sto punto teniamo la size di 25 k che è fattibile in tempi ragionevoli, ora ho provato ad aumentarei valori di n1 e n2
    

- slurm-418607
    - 1 master, 3 slave
    - 25k points
    - n1: 2, n2: 2
    - batchsize: 1000
    - iter train: 100
    - init time: 831.74
    - train time: 22623.68479204178 

- slurm-418611
    - 1 master, 3 slave
    - 25k points
    - n1: 2, n2: 4
    - batchsize: 1000
    - iter train: 100
    - init time: 425.02 
    - train time: 15993.591304540634

- slurm-418685
    - 1 master, 3 slave
    - 25k points
    - n1: 4, n2: 4
    - batchsize: 1000
    - iter train: 100
    - init time: 914.061 
    - train time: 66663.95424890518


- slurm-418804
    - 1 master, 3 slave
    - 25k points
    - n1: 1, n2: 4
    - batchsize: 1000
    - iter train: 100
    - init time: 852 
    - train time : 44191.26421666145

- slurm-....
    - 1 master, 3 slave
    - 25k points
    - n1: 4, n2: 1
    - batchsize: 1000
    - iter train: 100
    - init time: 
    - train time:

- slurm-........
    - 1 master, 3 slave
    - 25k points
    - n1: 3, n2: 3
    - batchsize: 1000
    - iter train: 100
    - init time:  
    - train time:
