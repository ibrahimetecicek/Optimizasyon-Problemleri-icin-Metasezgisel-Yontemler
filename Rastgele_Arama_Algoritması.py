# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:04:51 2021

@author: İbrahim Mete Çiçek
"""

# Kullanılan Kütüphaneler
import numpy as np

# en basit meta-sezgisel arama algoritması
def RS(f, init, nbr, its, stop=None):
    """
   f: amaç fonksiyonu X -> R (burada X arama alanıdır)
     init: rastgele X öğesini veren işlev
     nbr: x girişinin bir komşusunu veren X -> X fonksiyonu
     its: yineleme sayısı, yani uygunluk değerlendirme bütçesi
     stop: sonlandırma kriteri (X, R) -> bool
     dönüş: şimdiye kadarki en iyi x
    
     Bu sürümde, bir geçmişini saklıyor ve iade ediyoruz
     en iyi objektif değerler; objektif değerlendirmeleri israf etmekten kaçınırız;
     fesih kriterine izin veriyoruz.
    """
    history = [] # tarih oluştur
    x = init()
    fx = f(x) # fx mevcut en iyi noktanın f saklar
    for i in range(its):
        xnew = init()
        fxnew = f(xnew) # f'yi yeniden hesaplamaktan kaçınma
        if fxnew < fx: 
            x = xnew
            fx = fxnew
        history.append((i, fx)) # geçmişi kaydet
        if stop is not None and stop(x, fx): # bir fesih koşulu
            break
    return x, np.array(history) # dönüş geçmişi