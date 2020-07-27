# Bilgisayar Projesi ---->  ARA-PROJE  
# 2019 2020 Güz Dönemi
# DERİN ÖĞRENME YÖNTEMLERİ İLE MARKA İÇEREN NESNELERİN TESPİTİ VE BULANIKLAŞTIRILMASI

Projemizde kullandığımız OpenLogo veri seti ve detaylı açıklamaları aşağıdaki linktedir.
	https://qmul-openlogo.github.io/

Veri setini indirmek için aşağıdaki bağlantı kullanılabilir.
	https://drive.google.com/file/d/1p1BWofDJOKXqCtO0JPT5VyuIPOsuxOuj/view
  
  Asagidaki linkten resimleri labellamak için kullanilan tool indirilebilir.
https://github.com/tzutalin/labelImg

opencv_python==3.4.3.18
numpy==1.15.1


Projenin çalışması için bilgisayarınıza kurmanız gereken programlar:

-Anaconda3-5.2.0-Windows-x86_64

-cuda_9.0.176_win10

-cudnn-9.0-windows10-x64-v7

-TensorFlow kurulumu için aşağıdaki videodaki adımları takip ediniz..
https://www.youtube.com/watch?v=RplXYjxgZbw

-Bilgisayarınızda C diskinde tensorflow1 diye yeni bir klasör açınız.

-https://github.com/tensorflow/models bu linkteki dosyaları rar olarak indirip rarı çıkartınız ve model
dosyasını C diskinde açtığınız tensorflow1 klasörünün içine atınız.

-http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
yukarıdaki linkte bulunan dosyayı indirip C:\tensorflow1\models\research\object_detection klasörüne kopyalayınız.

-Son olarak aşağıdaki bağlantıdaki linkten adımları takip ederek gerekli kütüphane ve programları kurabilirsiniz.
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn

-Projenin kolay kurulabilmesi için aşağıdaki videoyu takip edebilirsiniz.
https://www.youtube.com/watch?v=Rgpfk6eYxJA

Eğitilmiş modelin boyutu 153 mb olduğu için bu dosyaya koyulamamıştır.Aşağıdaki linkten indirebilirsiniz.
https://s2.dosya.tc/server12/zga65g/inference_graph.rar.html

Bütün kurulumları tamamaladıktan sonra sırasıyla aşağıdaki adımları uygulayınız:
1-Anaonda prompt ekranını açınız

2-Activate tensorflow1 yazarak enter tuşuna basınız

3-cd C:\tensorflow1\models\research\object_detection yazarak dizine gidiniz.

4-python Object_detection_video.py yazarak projemizi çalıştırmış olursunuz

Bir arayüz açılır ve video seçmeniz beklenir.Video seçtikten sonra sistem videdoyu başlatır ve tespit ettiği markaları bulanıklaştırır.Bulanıklaştırılmış video
masaüstüne output.avi olarak çıkartılır ve işlem başarıyla tamamlanmış olur.
