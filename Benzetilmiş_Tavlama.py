# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:04:51 2021

@author: İbrahim Mete Çiçek
"""
# Kullanılan Kütüphaneler
import math
import numpy as np
import random


def SA(f, init, nbr, T, alpha, maxits):
    """Tavlama simülasyonu. Minimize ettiğimizi varsayalım.
     Şimdiye kadarki en iyi x'i ve f değerini döndür.

     Başlangıç sıcaklığı T ve bozunma faktörü alfa değerini geç.

     T, her adımda T * = alfa ile bozulur.
    """
    x = init() # rastgele bir ilk çözüm üretmek
    fx = f(x)
    bestx = x
    bestfx = fx
    history = [] # tarih yaratmak

    for i in range(1, maxits):
        xnew = nbr(x) # x'in bir komşusunu oluştur
        fxnew = f(xnew)
        
        # "kabul et" xnew eğer daha iyiyse VEYA daha kötüsü de bir
        # * Ne kadar kötü * ile ilgili küçük olasılık varsaymak.
        # Maksimize etmiyoruz, küçültüyoruz.
        if fxnew < fx or random.random() < math.exp((fx - fxnew) / T):
            x = xnew
            fx = fxnew

            # aynı zamanda şimdiye kadarki en iyiyi koruduğunuzdan emin olun x
            if fxnew < bestfx:
                bestx = x
                bestfx = fx
            
        T *= alpha # sıcaklık azalır
        #print(i, fx, T)
        history.append((i, fx)) # geçmişi kaydet
    return bestx, bestfx,np.array(history)