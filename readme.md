# This code is not good, so please don't look at it.

## Upgrades necessari

## Version 1

- Gli script per lanciare su kondo dovrebbero controllare che non ci
  sia un altro script con lo stesso nome.

## Version 2

- bruna10 dovrebbe essere in files esterni.

- Misure gradienti (al momento li misura un programma a parte).

- Misura correlazioni per layer.

- Misurare Loss/Accuracy a tw, in modo da fare solo un ciclo di
  misure, ora che si misura anche il gradiente.

- tw deve andare fino a tmax, non fino a tmax/2, e per tw>tmax
  semplicemente si fanno meno t.

- ResNet

- Relegare qualche funzione a moduli esterni (tipo i tempi),
  accorciare la lunghezza del main, fare qualche classe/nested method

- Calcolare gli istogrammi con pytorch invece che non numpy

- tutte le osservabili devono essere scritte su disco "on the run"


