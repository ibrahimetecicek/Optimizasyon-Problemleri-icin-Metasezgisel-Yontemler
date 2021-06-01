# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:04:51 2021

@author: İbrahim Mete Çiçek
"""
# Kullanılan Kütüphaneler
import numpy as np
import random

def real_gaussian_init(n):
    return np.random.normal(size=n)

def real_gaussian_nbr(x):
    delta = 0.1
    x = x.copy()
    # Gaussian
    x = x + delta * np.random.randn(len(x))
    return x

def stats(gen, popfit):
    # nesil numarasını döndürme
    # değerlendirilen bireylerin yüzdesi
    return gen, (gen+1) * len(popfit), np.min(popfit), np.mean(popfit), np.median(popfit), np.max(popfit), np.std(popfit)
   

# bu GA en aza indirmek içindir
def GA(f, init, nbr, crossover, select, popsize, ngens, pmut):
    history = []
    # ilk nüfusu oluşturun, kondisyonu değerlendirin, istatistikleri yazdırın
    pop = [init() for _ in range(popsize)]
    popfit = [f(x) for x in pop]
    history.append(stats(0, popfit))
    for gen in range(1, ngens):
        # boş yeni bir popülasyon yapın
        newpop = []
        # elitizm
        bestidx = min(range(popsize), key=lambda i: popfit[i])
        best = pop[bestidx]
        newpop.append(best)
        while len(newpop) < popsize:
            # seçme ve geçiş
            p1 = select(pop, popfit)
            p2 = select(pop, popfit)
            c1, c2 = crossover(p1, p2)
            # mutasyonu bireylerin yalnızca bir kısmına uygulayın
            if random.random() < pmut:
                c1 = nbr(c1)
            if random.random() < pmut:
                c2 = nbr(c2)
            # yeni bireyleri popülasyona ekleyin
            newpop.append(c1)
           # bu boyutunda yeni bir pop oluşturmayacağımızdan emin olun (popsize + 1) -
           # elitizm 1'i kopyaladığı için buna neden olabilir
            if len(newpop) < popsize:
                newpop.append(c2)
        # eski popülasyonun üzerine yeni, değerlendir ve istatistiklerle yaz
        pop = newpop
        popfit = [f(x) for x in pop]
        history.append(stats(gen, popfit))
    bestidx = np.argmin(popfit)
    return popfit[bestidx], pop[bestidx],np.array(history)

 
def tournament_select(pop, popfit, size):
     # Aynı kişi için f'yi birden çok kez yeniden hesaplamaktan kaçınmak için
     # fitness değerlendirmesini ana döngüye koyun ve sonucu
     # popfit. Bunu buradan iletiyoruz. Şimdi adaylar sadece
     # popülasyondaki bireyleri temsil eden endeksler.
    candidates = random.sample(list(range(len(pop))), size)
    # Kazanan, minimum kondisyona sahip bireyin endeksidir.
    winner = min(candidates, key=lambda c: popfit[c])
    return pop[winner]

def uniform_crossover(p1, p2):
    c1, c2 = [], []
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1.append(p1[i]); c2.append(p2[i])
        else:
            c1.append(p2[i]); c2.append(p1[i])
    return np.array(c1), np.array(c2)
