# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:04:51 2021

@author: İbrahim Mete Çiçek
"""
# Kullanılan Kütüphaneler
import numpy as np


"""Geç kabul edilen tepe tırmanışı (LAHC), stokastik bir tepe tırmanışıdır
Bykov tarafından önerilen bir tarih mekanizmasına sahip algoritma
[http://www.cs.nott.ac.uk/~yxb/LAHC/LAHC-TR.pdf].

Tarih mekanizması çok basit ama bazı alanlarda öyle görünüyor
yokuş tırmanmaya kıyasla kayda değer bir performans artışı sağlar
kendisi ve diğer buluşsal yöntemler. En büyük avantajı basitliğidir:
Birçok alternatif yönteme kıyasla ayarlanacak daha az parametre.

Standart stokastik tepe tırmanışında, yeni bir
önerilen nokta (mutasyonla oluşturulmuş) eğer bu nokta ya da kadar iyiyse
şu anki noktadan daha iyi.

LAHC'de, yeni nokta kadar iyi veya daha iyiyse hareketi kabul ediyoruz
ondan L adım önce karşılaştık. L tek yeni parametredir
tepe tırmanmaya kıyasla: tarihin uzunluğunu temsil eder.

"""


"""
Bu, Bykov ve Burke'ün LAHC sözde kodudur.

Bir ilk çözüm üretin
İlk maliyet fonksiyonunu hesaplayın C (s)
Lfa belirtin
{0 ... Lfa-1} f_k'deki tüm k için: = C (s)
İlk yineleme I = 0;
Seçilen bir durdurma durumuna kadar yapın
     Aday bir çözüm oluşturun *
     Maliyet fonksiyonunu hesaplayın C (s *)
     v: = Ben mod Lfa
     C (s *) <= fv veya C (s *) <= C (s) ise
     Ardından adayı kabul edin (s: = s *)
     Aksi takdirde adayı reddedin (s: = s)
     Geçerli maliyeti uygunluk dizisine ekle fv: = C (s)
     Yineleme numarasını artırın I: = I + 1

"""

# L geçmiş uzunluğudur
# n değerlendirmelerin bütçesidir
# C maliyet fonksiyonudur
# init, başlangıçtaki bir kişiyi oluşturan bir işlevdir
# nbr bir komşuluk işlevidir

def LAHC(L, n, C, init, nbr):
    s = init()            # ilk çözüm
    Cs = C(s)             # mevcut çözümün maliyeti
    best = s              # en iyi çözüm
    Cbest = Cs            # en iyinin maliyeti
    f = [Cs] * L          # başlangıç geçmişi
    history = []          # tarih oluştur
    # print(0, Cbest, best)
    for I in range(1, n): # yineleme sayısı
        s_ = nbr(s)       # aday çözüm
        Cs_ = C(s_)       # adayın maliyeti
        if Cs_ < Cbest:   # küçültme
            best = s_     # en iyi güncelleme
            Cbest = Cs_
            history.append((I, Cbest)) # geçmişi kaydet
            # print(I, Cbest, best)
        v = I % L         # v dizinler I döngüsel olarak
        if Cs_ <= f[v] or Cs_ <= Cs:
            s = s_        # adayı kabul et
            Cs = Cs_      # (aksi takdirde reddet)
        f[v] = Cs         # döngü geçmişini güncelle
    return best, Cbest,np.array(history)