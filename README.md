# Optimizasyon-Problemleri-icin-Metasezgisel-Yontemler

Problemin Açıklanması

Bu final projesi için meyve toplama problemini (Facility location problem) düşündüm. Problemde bir fabrikanın büyük meyve bahçesi var ve bu bahçede x koordinatı, y koordinatı ve toplam meyve sayısı kullanılarak temsil edilen 2 boyutlu uzaya yayılan n ağaç (n = 10) vardır. Yapılan iş miktarının minimum olması için 6 sepet (12 gerçek değerli karar değişkeni) kullanmamız gerekir. Amacımız ağaçtan ağaca giderken yapılan işi yani hareketin cost’unu minimum olmasını sağlayan yolu seçmek diğer bir deyişle minimum cost’u elde etmektir.

Bu projede Yapay Arı Kolonisi (Artificial Bee Colony), Tepe Tırmanma (Hill Climbing), Benzetilmiş Tavlama (Simulated Annealing), Geç Kabul Edilen Tepe Tırmanma (Late Accepted Hill Climbing), Genetik Algoritması (Genetic Algorithm) gibi farklı optimizasyon algoritmaları sıfırdan uygulanmaktadır.
Kovaryans Matris adaptasyonu ve Parçacık Sürüsü Algoritması (Particle Swarm Algorithm) gibi algoritmalar için harici Python kütüphaneleri kullanılmıştır. Son olarak, her algoritmanın rastgele ağaç üzerinde beş kez çalıştığı ve amaç fonksiyonunun ortalama ve standart sapmasının sunulduğu bir algoritma karşılaştırması tablo halinde sunulmuştur.
