
import numpy as np
import random
import matplotlib.pyplot as plt
from Tepe_Tırmanma import HC
from Colony import ColonyAlgo


# İTHALAT VERİ SETİ (Ağaçların konumu ve üzerlerindeki meyve miktarı)
filename = 'meyve_toplama.txt'
m = 6 # sepet sayısı
data = np.genfromtxt(filename)
T = data[:, 0:2] # x, y ağaçların yerleri
weight = data[:, 2] # ağaç başına meyve miktarı
n = T.shape[0]

def dist(a, b):
    # Öklid uzaklığı
    return np.sqrt(np.sum((a - b) ** 2))


def facility_location(x):
    x = np.array(x)
  # amaç: en yakın çöp kutusuna olan ağırlıklı mesafeyi en aza indirgemek, ağaçlar arasında toplamak.
    bins = x.reshape((int(len(x)/2), 2))
    total_dist = 0.0
    for t in range(n):
        tree = T[t]
        nearest = min(dist(tree, b) for b in bins)
        total_dist += weight[t] * nearest
    return total_dist


# GENEL BAŞLANGIÇ VE KOMŞU FONKSİYONLARI
def real_init(n):
    return np.random.random(n)

def real_nbr(x):
    delta = 0.5
    x = x.copy()
    i = random.randrange(len(x))
  # [-delta, delta] aralığında küçük bir gerçek sabit ekleyin
    x[i] = x[i] + (2 * delta * np.random.random() - delta)
    return x

def facility_run_abc(food_source=6,maxints=5000,plot=False):
    n=12
    f = lambda x: facility_location(x) 
    bestx, bestfx,history =ColonyAlgo(f,
                      lambda: real_init(n),
                      real_nbr,
                      food_source,
                      maxints)

    if(plot):
        plt.plot([pl[0] for pl in history], [pl[1] for pl in history])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        titleStr = "Colony food_source = "+str(food_source)
        plt.title(titleStr)
        plt.show()
    return bestx,bestfx    

"""
# Test için Tek Çalışma
X = facility_run_Simulated_Annealing()
facility_location(X)
"""
# Çizim amaçlı
# X = facility_run_Simulated_Annealing(T=1,alpha=0.5,maxints=2000,plot=True)

for food_source in range(6,10):
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_abc(food_source=6,maxints=300,plot= True)
        fvalues.append(fbest)

    fvalues = np.array(fvalues)   
    print(f"\n\n ABC için amaç fonksiyonu değerleri (#bees = {food_source}) : ",fvalues)
    print(f"ABC için ortalama amaç fonksiyonu değeri (#bees  = {food_source}) : ",np.mean(fvalues))
    print(f"ABC için amaç fonksiyonu için standart sapma (#bees  = {food_source}) : ",np.std(fvalues))