# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:04:51 2021

@author: İbrahim Mete Çiçek
"""

# Kullanılan Kütüphaneler
import numpy as np
import random
import matplotlib.pyplot as plt
from Tepe_Tırmanma import HC
from Benzetilmiş_Tavlama import SA
from Geç_Kabul_Edilen_Tepe_Tırmanma import LAHC
from Genetik_Algoritması import GA,tournament_select,uniform_crossover,real_gaussian_init,real_gaussian_nbr
from Rastgele_Arama_Algoritması import RS
import cma
from pyswarm import pso
import itertools
from En_İyi_Komşuluk import BestNeighborsAlgo



# İTHALAT VERİ KÜMESİ (Ağaçların yeri ve üzerlerindeki meyve miktarı)
filename = 'meyve_toplama.txt'
m = 6 # sepet sayısı
data = np.genfromtxt(filename)
T = data[:, 0:2] # x, y ağaçların yerleri
weight = data[:, 2] # ağaç başına meyve miktarı
n = T.shape[0]

def dist(a, b):
    # Öklid mesafesi
    return np.sqrt(np.sum((a - b) ** 2))


def facility_location(x):
    x = np.array(x)
    # amaç: Ağaçlar arasında toplanan en yakın sepete olan ağırlıklı mesafeyi en aza indirme.
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
    delta = 0.3
    x = x.copy()
    i = random.randrange(len(x))
    # aralığa küçük bir gerçek sabit ekleyin [-delta, delta]
    x[i] = x[i] + (2 * delta * np.random.random() - delta)
    return x

######################################
# RASGELE ARAMA İÇİN YARDIMCI FONKSİYON #
######################################
def facility_run_RandomSearch(its=10000,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    stop = lambda i, fx: abs(fx) < 0.00001
    x,history = RS(f, lambda: real_init(n), real_nbr, its, stop=stop)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        plt.title("Random seacrh")
        plt.show()
    fbest = facility_location(x)
    return x,fbest

# Çizim amaçlı
# X = facility_run_RandomSearch(its=50000,plot=True)
seedList = [i for i in range(5)]
fvalues=[]
for s in seedList:
    random.seed(s)
    x,fbest = facility_run_RandomSearch(its=50000)
    fvalues.append(fbest)
    
fvalues = np.array(fvalues)   
print("Rastgele arama için amaç fonksiyon değerleri şunlardır: ",fvalues)
print("Rastgele arama için ortalama amaç fonksiyon değeri: ",np.mean(fvalues))
print("Rastgele arama için amaç işlevinin standart sapması şudur: ",np.std(fvalues))

################################################
# TEPE TIRMANMA FONKSİYONU İÇİN YARDIMCI FONKSİYON #
################################################
def facility_run_Hill_Climbing(its=10000,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    stop = lambda i, fx: abs(fx) < 0.00001
    x,history = HC(f, lambda: real_init(n), real_nbr, its, stop=stop)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        plt.title("Hill Climbing")
        plt.show()
    fbest = facility_location(x)
    return x,fbest

"""  
# Single Run for testing    
X = facility_run_Hill_Climbing()
facility_location(X[0])
"""
# For plotting purpose
# X = facility_run_Hill_Climbing(its=2000,plot=True)

seedList = [i for i in range(5)]
fvalues=[]
for s in seedList:
    random.seed(s)
    x,fbest = facility_run_Hill_Climbing(its=50000)
    fvalues.append(fbest)

fvalues = np.array(fvalues)   
print("Tepe Tırmanışı için amaç fonksiyon değerleri şunlardır: ",fvalues)
print("Hill Climbing için ortalama amaç fonksiyon değeri: ",np.mean(fvalues))
print("Tepe Tırmanışı için amaç işlevinin standart sapması şu şekildedir: ",np.std(fvalues))

############################################
# BENZETİLMİŞ TAVLAMA İÇİN FAYDALI FONKSİYON #
############################################
def facility_run_Simulated_Annealing(T=1,alpha=0.999,maxints=5000,plot=False):
    n=12
    f = lambda x: facility_location(x) 
    bestx, bestfx,history =SA(f,
                      lambda: real_init(n),
                      real_nbr,
                      T,
                      alpha,
                      maxints)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        titleStr = "Simulated Annealing T = "+str(T)+" alpha = "+str(alpha)
        plt.title(titleStr)
        plt.show()
    return bestx,bestfx    

"""
# Single Run for testing 
X = facility_run_Simulated_Annealing()
facility_location(X)
"""
# Çizim amaçlı
# X = facility_run_Simulated_Annealing(T=1,alpha=0.5,maxints=2000,plot=True)

Tvalues = [1,2]
alphaValues = [0.5,0.9]
for temp,alpha in itertools.product(Tvalues,alphaValues):
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_Simulated_Annealing(T=temp,alpha=alpha,maxints=50000)
        fvalues.append(fbest)

    fvalues = np.array(fvalues)   
    print(f"\n\nBenzetilmiş Tavlama için amaç fonksiyon değerleri (T = {temp} , alpha ={alpha} ) : ",fvalues)
    print(f"Benzetilmiş Tavlama için ortalama amaç fonksiyon değeri (T = {temp} , alpha ={alpha} ) : ",np.mean(fvalues))
    print(f"Benzetilmiş Tavlama için amaç işlevi için standart sapma (T = {temp} , alpha ={alpha} ) : ",np.std(fvalues))

####################################################
# GEÇ KABUL EDİLEN TEPE TIRMANIŞI İÇİN HİZMET FONKSİYONU #
####################################################
def facility_run_LAHC(L=10,maxiter=10000,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    best, Cbest,history = LAHC(L, maxiter, f, lambda: real_init(n), real_nbr)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        titleStr = "LAHC L = "+str(L)
        plt.title(titleStr)
        plt.show()
    return best,Cbest

"""
# Test için Tek Çalıştırma
bestx,bestfx = facility_run_LAHC(L=2) 
"""
# Çizim amaçlı
# X = facility_run_LAHC(L=10,maxiter=3000,plot=True)

Lvalues = [2,10,20]
for L in Lvalues:
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_LAHC(L=L,maxiter=50000)
        fvalues.append(fbest)
        
    fvalues = np.array(fvalues)   
    print(f"\n\nLAHC için amaç fonksiyonu değerleri (L = {L} ) : ",fvalues)
    print(f"LAHC için ortalama amaç fonksiyon değeri (L = {L} ) : ",np.mean(fvalues))
    print(f"LAHC için amaç işlevi için standart sapma (L = {L} ) : ",np.std(fvalues))


##########################################
# GENETİK ALGORİTMA İÇİN FAYDALI FONKSİYON #
##########################################
def facility_run_Genetic_Algorithm(popsize = 100,ngens = 100,pmut = 0.1,tsize = 2,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    bestf, best, h = GA(f,
                    lambda: real_gaussian_init(n),
                    real_gaussian_nbr,
                    uniform_crossover,
                    lambda pop, popfit: tournament_select(pop, popfit, tsize),
                    popsize,
                    ngens,
                    pmut
                    )
    if(plot):
        plt.plot(h[:, 1], h[:, 2])
        plt.xlabel("Iterations");plt.ylabel("Fitness")
        titleStr = "Genetic Algo P = "+str(popsize)+" G = "+str(ngens)+" M = "+str(pmut)+" T = "+str(tsize)
        plt.title(titleStr)
        plt.show()
        plt.close()
    
        # plot std birey sayısına göre uyumu
        plt.plot(h[:, 1], h[:, -1])
        plt.xlabel("Iterations");plt.ylabel("Standard deviation (fitness)")
        plt.title(titleStr)
        plt.show()
    return best,bestf

"""
# Test için Tek Çalıştırma
X = facility_run_Genetic_Algorithm(popsize = 100,ngens = 100,pmut = 0.1,tsize = 2)
facility_location(X[0])
"""
# For plotting purpose
# X = facility_run_Genetic_Algorithm(popsize = 200,ngens = 250,pmut = 0.2,tsize = 2,plot=True)

populationSize = [100,200]
mutationRatio = [0.1,0.2]
tournamentSize = [2]

for pop,pmut,tsize in itertools.product(populationSize,mutationRatio,tournamentSize):
    seedList = [i for i in range(5)]
    fvalues=[]
    # 50000 bütçelik bir fitness değerlendirme bütçesi dikkate alındığında = populationSize * noOfGeneration
    gen = int(50000/pop)
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_Genetic_Algorithm(popsize = pop,ngens = gen,pmut = pmut,tsize = tsize)
        fvalues.append(fbest)
    fvalues = np.array(fvalues)   
    print(f"\n\nGenetik Algoritma için amaç fonksiyon değerleri ( Popülasyon boyutu : {pop} ,  Gen : {gen} , Mutasyon oranı :{pmut} , Turnuva boyutu : {tsize} : ",fvalues)
    print(f"Genetik Algoritma için ortalama amaç fonksiyon değeri ( Popülasyon boyutu : {pop} ,  Gen : {gen} , Mutasyon oranı :{pmut} , Turnuva boyutu : {tsize} : ",np.mean(fvalues))
    print(f"Genetik Algoritma için amaç işlevi için standart sapma ( Popülasyon boyutu : {pop} ,  Gen : {gen} , Mutasyon oranı :{pmut} , Turnuva boyutu : {tsize} : ",np.std(fvalues))


#####################################################
# KOVARYANS MATRİSİ ADAPTASYONU İÇİN FAYDALI FONKSİYON #
#####################################################
def estimate_full(pop):
    # "pop" daki örneklerin dağılımını tahmin edin: model
    # ortalama (bir vektör) ve tam kovaryans matrisinden oluşur.
    mu = np.mean(pop, axis=0)
    sigma = np.cov(pop, rowvar=False)
    return mu, sigma

def facility_run_CMA(popsize=100):
    # Reference : http://cma.gforge.inria.fr/html-pythoncma/frames.html
    # The CMA Evolution Strategy: A Tutorial : https://arxiv.org/pdf/1604.00772.pdf (From: Nikolaus Hansen)
    n = 12
    f = lambda x: facility_location(x) 
    pop = [real_init(n) for i in range(popsize)]
    mu,sigma = estimate_full(pop)
    es=cma.CMAEvolutionStrategy(mu,1,
                                {'bounds': [-np.inf, np.inf],
                                 'seed':234,
                                 'popsize':popsize,
                                 'maxiter':1000})
    es.optimize(f)    
    xbest, fbest, evals_best, evaluations, iterations, xfavorite, stds, stop = es.result
    return xbest,fbest

"""
# Single Run for testing 
X = facility_run_CMA()
facility_location(X[0])
"""

populationSize = [10,100,200]
for pop in populationSize:
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_CMA(popsize=pop)
        fvalues.append(fbest)
        
    fvalues = np.array(fvalues)   
    print(f"\n\nCMA için amaç fonksiyon değerleri ( Popülasyon boyutu : {pop} )  : ",fvalues)
    print(f"CMA için ortalama amaç fonksiyon değeri ( Popülasyon boyutu : {pop} )  : ",np.mean(fvalues))
    print(f"CMA için amaç işlevi için standart sapma ( Popülasyon boyutu : {pop} )  : ",np.std(fvalues))

#################################################
# PARÇACIK SÜRÜSÜ ALGORİTMASI İÇİN FAYDALI FONKSİYON #
#################################################
# Aşağıdaki bağlantıyı kullanarak PSO'yu sıfırdan uygulamayı denedim, ancak iyi sonuçlar vermedi (rastgele aramadan daha kötüsü)
# Sıfırdan PSO: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
# MALİYET İŞLEVLERİNİ TANIMLA

def facility_run_PSO(maxiter=1000,omega=0.5,phip =0.5,phig =0.5):
     # Referans: https://pythonhosted.org/pyswarm/ (Python'un PMO Uygulaması için PySwarm Kitaplığı)
     # maxiter: Sürü için aranacak maksimum yineleme sayısı (Varsayılan: 100)
     # omega: Parçacık hızı ölçekleme faktörü (Varsayılan: 0,5)
     # phip: Parçacığın en iyi bilinen konumundan uzakta arama yapmak için ölçekleme faktörü (Varsayılan: 0,5)
     # phig: Sürünün en iyi bilinen konumundan uzakta arama yapmak için ölçeklendirme faktörü (Varsayılan: 0,5)
    n = 12
    lb = [0 for i in range(n)] # Çiftlik menzilinde olmak (alt sınır)
    ub = [8 for i in range(n)] # Çiftlik menzilinde olmak (üst sınır)
    xoft,fopt=pso(facility_location,lb,ub,maxiter=maxiter)
    #xoft,fopt= pso(facility_location,lb,ub,omega=omega,phip=phip,phig=phig,maxiter=maxiter)
    return xoft,fopt

""" 
# Test için Tek Çalıştırma
X,f=facility_run_PSO()
"""
# PSO için temel formüller
# V = W*V + cp*rand()(pbest-x) + cg*rand()(gbest-x)
# X = X +V
W = [0.5,1]
cp = [0.5,0.9] # tam olarak formülün cp'si değil
cg = [0.5,0.9] # tam olarak formülün cg'si değil

for w,p,g in itertools.product(W,cp,cg):
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_PSO(omega=w,phip=cp,phig=cg)
        fvalues.append(fbest)
    fvalues = np.array(fvalues)   
    print(f"\n\nPSO için amaç fonksiyon değerleri ( w = {w} , cp ={p} , cg ={g} ) : ",fvalues)
    print(f"PSO için ortalama amaç fonksiyon değeri ( w = {w} , cp ={p} , cg ={g} ) : ",np.mean(fvalues))
    print(f"PSO için amaç işlevi için standart sapma ( w = {w} , cp ={p} , cg ={g} ) : ",np.std(fvalues))


#################################################################################
# EN İYİ KOMŞU ALGORİTMASI İÇİN FAYDALI FONKSİYON - YENİ ALGORİTMA OLUŞTURMA DENEMESİ #
#################################################################################
# Algoritma En_İyi_Komşuluk.py'de uygulanmaktadır
def facility_run_Best_Neighbors_Algo(its=50,c_hyperparam=3):
    n = 12
    f = lambda x: facility_location(x) 
    fbest = BestNeighborsAlgo(f, lambda: real_init(n), its,c=c_hyperparam) # Hiperparametre c'yi 3 olarak ayarlayın (iyi sonuçlar veriyor)
    return fbest


for c_h in [3,4,5,6]:
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        fbest =  facility_run_Best_Neighbors_Algo(its=50,c_hyperparam=c_h)
        fvalues.append(fbest)
    
    fvalues = np.array(fvalues)   
    print(f"\n\nEn İyi Komşular algoritması için amaç fonksiyon değerleri  ( c = {c_h} )  : ",fvalues)
    print(f"Best Neighbors algoritması için ortalama amaç fonksiyon değeri  ( c = {c_h} ) : ",np.mean(fvalues))
    print(f"En İyi Komşular algoritması için amaç işlevi için standart sapma  ( c = {c_h} ) : ",np.std(fvalues))
