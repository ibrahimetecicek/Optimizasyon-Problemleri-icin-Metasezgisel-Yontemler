# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:04:51 2021

@author: İbrahim Mete Çiçek
"""

# Kullanılan Kütüphaneler
import random
import collections

# N boyutlu giriş vektörü için n komşu oluşturur (uzaklık, orijinal giriş vektöründen rastgele)
def nearby(X,c):
    newList=list()
    X_copy = X.copy()
    for i in range(len(X)):
        X_copy[i] = X[i] + (c*random.random()) # c hiperparametredir
        newList.append(X_copy)
        X_copy = X.copy()
    return newList

# nearby([1,2,3,4,5],1)
  
# Not: Kod şu anda kötü yazılmış, bu hafta sonu onu daha iyi hale getirmeye çalışacak
def BestNeighborsAlgo(f,init,its,c=3,m=100):
    x = init() # ilk X ile başlayın
    domainList = list() # nüfusumuzu depolar (nüfusumuzu m, varsayılan olarak m100 ile sınırlayacağız, böylece nüfusumuz asla m'yi aşmaz)
    domainList.append(x)
    for i in range(its):
        newDomainList = list()
        for e in domainList:
            nearList = nearby(e,c) # etki alanındaki her x için Liste n boyutlu komşularını oluşturur
            newDomainList.extend(nearList) 
            newDomainList.append(e) # orijinal x'i de ekleyin
            
        D = dict()
      # bir hedef_değer diktesi oluşturun: x
        for d in newDomainList:
            D[f(d)]=d
          
        # Sözlüğü tuşlara göre sırala (amaç fonksiyonu)
        P = collections.OrderedDict(sorted(D.items()))
        size = min(len(P),m)
       # sonraki popülasyonu boş yap
        domainList = list()
        # sonraki etki alanı listemiz (popülasyon) yalnızca en iyi m vektörlerine sahip olacak (mimimizasyon durumunda en düşük hedef değerin bir)
        ip=0
        for p in P.values():
            ip=ip+1
            if ip>size:
                break
            domainList.append(p)
      # bu domainList sonraki nüfus olarak hareket edecek  
    Y = [f(y) for y in domainList]
    return min(Y)
