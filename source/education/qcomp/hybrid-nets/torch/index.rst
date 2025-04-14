============================================
PyTorch Nöral Ağına Kuantum Katmanı Eklemek
============================================

.. warning::
    Belgeyi okumaya başlamadan önce :ref:`qcomp_setup`'da anlatıldığı gibi içinde Pennylane ve Lightning-GPU kurulu bir konteyner oluşturduğunuzdan emin olun.
    
Gerekli kütüphaneleri yüklemek için bu konteyneri aşağıdaki gibi açınız 

.. code-block:: bash    
    apptainer shell --no-home --writable --fakeroot miniconda3-user
    # veya
    apptainer shell --no-home --writable miniconda3-user

ve sonrasında bu dokümanda bize gerekli olacak scikit-learn ve PyTorch kütüphanelerini aşağıdaki gibi yükleyiniz.

.. code-block:: bash

    pip install torch torchvision torchaudio 
    pip install scikit-learn

Daha sonrasında konteynerin imajını oluşturmak için:

.. dropdown:: :octicon:`codespaces;1.5em;secondary` Konteyner İmajı Oluşturma(Tıklayınız)
    :color: info

        .. tab-set::

            .. tab-item:: İş Gönderme

                .. code-block:: bash

                    sbatch build.slurm

            .. tab-item:: build.slurm

                .. code-block:: bash
            
                    #!/bin/bash
                    
                    #SBATCH --output=slurm-%j.out
                    #SBATCH --error=slurm-%j.err
                    #SBATCH --time=00:15:00
                    #SBATCH --job-name=build

                    #SBATCH -p debug
                    #SBATCH --partition=orfoz
                    #SBATCH --ntasks=1
                    #SBATCH --nodes=1
                    #SBATCH -C weka
                    #SBATCH --cpus-per-task=55

                    apptainer build miniconda3-user.sif miniconda3-user

Klasik Torch Katmanlarından oluşan bir nöral ağ oluşturma
==========================================================


Burada PennyLane kütüphanesindeki bir Quantum Node'un nasıl PyTorch kütüphanesindeki bir layer olarak eklendiğine yönelik bir örnek gösteriliyor. PyTorch hakkında detaylı bilgi için :ref:`pytorch_intro` eğitim materyaline bakabilirsiniz.

Aşağıdaki gibi kuantum katmanı içermeyen ikili sınıflandırmada kullanabileceğimiz 2 katmanlı bir network oluşturabiliriz.

.. code-block:: python

    import torch

    layer_1 = torch.nn.Linear(2, 2)
    layer_2 = torch.nn.Linear(2, 2)
    softmax = torch.nn.Softmax(dim=1)

    layers = [layer_1, layer_2, softmax]
    model = torch.nn.Sequential(*layers)

Şimdi bu katmanlı yapının içine nasıl kuantum katmanı da ekleyebileceğimizi göreceğiz.

Veri Kümesini Hazırlama
==================================

Burada kolay anlaşılması için basit bir veri kümesi olan ``scikit-learn`` içindeki ``make_moons`` veri kümesini kullanarak ikili sınıflandırma yapacağız.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_moons

    # Rastgele sayılar için tohum değerlerini belirleme
    torch.manual_seed(42)
    np.random.seed(42)

    X, y = make_moons(n_samples=200, noise=0.1)
    y_ = torch.unsqueeze(torch.tensor(y), 1)  # one-hot encoding ile kodlanmış etiketler
    y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)


Quantum Node Oluşturma
======================

Modelimizi GPU'da çalıştırabilmek için simülatör olarak ``lightning.gpu`` kullanıyoruz. Buradaki ``weights`` argümanı eğitilebilir ağırlık olarak kullanılacak.

.. code-block:: python

    import pennylane as qml

    n_qubits = 2
    dev = qml.device("lightning.gpu", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


Quantum Node'u PyTorch Katmanına Çevirme
=========================================

Bu işlem için Quantum Node'a argüman olarak gelen tüm eğitilebilir ağırlıkların şeklini belirtmemiz gerekiyor. Bu işlem için bir dictionary argüman isimlerini onların şekillerine map'leyen bir dictionary kullanabiliriz.

.. code-block:: python

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}

Bizim örneğimizdeki ``weights`` argümanının şekli (n_layers, n_qubits) olarak ``BasicEntanglerLayers()`` 'a aktarıldı. Dictionary'mizi oluşturduktan sonra kolay bir şekilde Quantum Node'umuzu bir Torch katmanına çevirebiliriz.


.. code-block:: python

    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

Sıralı Hibrit Model Oluşturma
==============================

Sayfanın en başındaki iki katmanlı network yapısının arasına kuantum katmanı eklenmiş halini aşağıdaki gibi oluşturabiliriz:

#. 2 nöronlu tamamen bağlı klasik katman
#. Bizim 2 kübitlik Quantum Node'dan çevirerek oluşturduğumuz kuantum katman
#. Başka bir tane daha 2 nöronlu tamamen bağlı klasik katman
#. Olasılık vektörüne çevirmek için ``softmax`` aktivasyonu


.. code-block:: python

    clayer_1 = torch.nn.Linear(2, 2)
    clayer_2 = torch.nn.Linear(2, 2)
    softmax = torch.nn.Softmax(dim=1)
    layers = [clayer_1, qlayer, clayer_2, softmax]
    model = torch.nn.Sequential(*layers)

Burada clayer'lar klasik katmanları qlayer ise kuantum katmanını gösteriyor. Böylece iki klasik katman arasına bir kuantum katmanını eklemiş olduk.

Sıralı Modeli Eğitme
=====================

Biz bu örnek için standart ``SGD optimizer`` 'ını ve ``mean absolute error`` loss function'ını kullanarak modelimizi eğiteceğiz ancak bu seçimlerin farklı kombinasyonları da tabii ki kullanılabilir.

.. code-block:: python

    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    loss = torch.nn.L1Loss()

    X = torch.tensor(X, requires_grad=True).float()
    y_hot = y_hot.float()

    batch_size = 5
    batches = 200 // batch_size

    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
    )

    epochs = 6

    for epoch in range(epochs):

        running_loss = 0

        for xs, ys in data_loader:
            opt.zero_grad()

            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()

            opt.step()

            running_loss += loss_evaluated

        avg_loss = running_loss / batches
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred = model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy * 100}%")

Sıralı Hibrit Modeli Kuyruğa Gönderme
=====================================
Aşağıdaki gibi bir slurm betiği hazırlayarak yukarıda oluşturduğumuz sıralı hibrit modeli kuyruğa gönderebiliriz.

.. dropdown:: :octicon:`codespaces;1.5em;secondary` Sıralı Model (Tıklayınız)
    :color: info

        .. tab-set::

            .. tab-item:: İş Gönderme

                .. code-block:: bash

                    sbatch sequential_qnn.slurm

            .. tab-item:: sequential_qnn.slurm

                .. code-block:: bash
            
                    #!/bin/bash

                    #SBATCH --output=slurm-%j.out
                    #SBATCH --error=slurm-%j.err
                    #SBATCH --time=00:15:00
                    #SBATCH --job-name=sequential_qnn

                    #SBATCH -p debug
                    #SBATCH --partition=akya-cuda
                    #SBATCH --gres=gpu:1
                    #SBATCH --ntasks=1
                    #SBATCH --nodes=1
                    #SBATCH --cpus-per-task=10

                    apptainer exec --nv miniconda3-user.sif python sequential_qnn.py

            .. tab-item:: sequential_qnn.py
                
                .. code-block:: python

                    import torch
                    import pennylane as qml
                    import numpy as np
                    from sklearn.datasets import make_moons


                    # Rastgele sayılar için tohum değerlerini belirleme
                    torch.manual_seed(42)
                    np.random.seed(42)

                    X, y = make_moons(n_samples=200, noise=0.1)
                    y_ = torch.unsqueeze(torch.tensor(y), 1)  # one-hot encoding ile kodlanmış etiketler
                    y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

                    n_qubits = 2
                    dev = qml.device("lightning.gpu", wires=n_qubits)

                    # Quantum Node oluşturma
                    @qml.qnode(dev)
                    def qnode(inputs, weights):
                        qml.AngleEmbedding(inputs, wires=range(n_qubits))
                        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
                        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

                    n_layers = 6
                    weight_shapes = {"weights": (n_layers, n_qubits)}

                    # Quantum Node'u kuantum katmanına çevirme
                    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

                    # Modeli oluşturma
                    clayer_1 = torch.nn.Linear(2, 2)
                    clayer_2 = torch.nn.Linear(2, 2)
                    softmax = torch.nn.Softmax(dim=1)
                    layers = [clayer_1, qlayer, clayer_2, softmax]
                    model = torch.nn.Sequential(*layers)

                    # Modeli eğitme
                    opt = torch.optim.SGD(model.parameters(), lr=0.2)
                    loss = torch.nn.L1Loss()

                    X = torch.tensor(X, requires_grad=True).float()
                    y_hot = y_hot.float()

                    batch_size = 5
                    batches = 200 // batch_size

                    data_loader = torch.utils.data.DataLoader(
                        list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
                    )

                    epochs = 6

                    for epoch in range(epochs):

                        running_loss = 0

                        for xs, ys in data_loader:
                            opt.zero_grad()

                            loss_evaluated = loss(model(xs), ys)
                            loss_evaluated.backward()

                            opt.step()

                            running_loss += loss_evaluated

                        avg_loss = running_loss / batches
                        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

                    y_pred = model(X)
                    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

                    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
                    accuracy = sum(correct) / len(correct)
                    print(f"Accuracy: {accuracy * 100}%")


Sıralı Olmayan Hibrit Model Oluşturma
=====================================

Sıralı katmanlar kullanarak oluşturulan modeller yaygın ve işlevli olsa da bazı durumlarda biz modelin nasıl inşa edildiği hakkında daha fazla kontrole sahip olmak isteriz. Örneğin, bazı durumlarda bi katmandaki çıktıları birden fazla katmana dağıtmak isteyebiliriz. Bunun için sıralı olmayan modelleri kullanabiliriz.


Biz aşağıdaki yapıdaki bir hibrit model oluşturmak istiyoruz:

#. 4 nöronlu tamamen bağlı klasik katman
#. Önceki klasik katmanın ilk 2 nöronuyla bağlı 2 kübitlik kuantum katman
#. Önceki klasik katmanın son 2 nöronuyla bağlı 2 kübitlik kuantum katman
#. Önceki kuantum katmanlarının kombinasyonundan 4 boyutlu bir girdi alan 2 nöronlu tamamen bağlı klasik katman
#. Olasılık vektörüne çevirmek için ``softmax`` aktivasyonu

Bunu başarmak için ``torch.nn.Module`` 'ün bir alt sınıfını yaratarak ``forward()`` methodunu override etmeliyiz.

.. code-block:: python

    class HybridModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clayer_1 = torch.nn.Linear(2, 4)
            self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
            self.qlayer_2 = qml.qnn.TorchLayer(qnode, weight_shapes)
            self.clayer_2 = torch.nn.Linear(4, 2)
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x):
            x = self.clayer_1(x)
            x_1, x_2 = torch.split(x, 2, dim=1)
            x_1 = self.qlayer_1(x_1)
            x_2 = self.qlayer_2(x_2)
            x = torch.cat([x_1, x_2], axis=1)
            x = self.clayer_2(x)
            return self.softmax(x)

    model = HybridModel()


Sıralı Olmayan Modeli Eğitme
=============================

Biz bu örnek için de standart ``SGD optimizer`` 'ını ve ``mean absolute error`` loss function'ını kullanarak modelimizi eğiteceğiz ancak bu seçimlerin farklı kombinasyonları da tabii ki kullanılabilir.

.. code-block:: python

    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    loss = torch.nn.L1Loss()

    X = torch.tensor(X, requires_grad=True).float()
    y_hot = y_hot.float()

    batch_size = 5
    batches = 200 // batch_size

    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
    )

    epochs = 6

    for epoch in range(epochs):

        running_loss = 0

        for xs, ys in data_loader:
            opt.zero_grad()

            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()

            opt.step()

            running_loss += loss_evaluated

        avg_loss = running_loss / batches
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred = model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy * 100}%")


Sıralı Olmayan Modeli Kuyruğa Gönderme
======================================
Benzer şekilde sıralı olmayan hibrit model için de aşağıdaki gibi bir slurm betiği hazırlayarak kuyruğa gönderebiliriz.

.. dropdown:: :octicon:`codespaces;1.5em;secondary` Sıralı Olmayan Model (Tıklayınız)
    :color: info

        .. tab-set::

            .. tab-item:: İş Gönderme

                .. code-block:: bash

                    sbatch nonsequential_qnn.slurm

            .. tab-item:: nonsequential_qnn.slurm

                .. code-block:: bash
            
                    #!/bin/bash

                    #SBATCH --output=slurm-%j.out
                    #SBATCH --error=slurm-%j.err
                    #SBATCH --time=00:15:00
                    #SBATCH --job-name=nonsequential_qnn

                    #SBATCH -p debug
                    #SBATCH --partition=akya-cuda
                    #SBATCH --gres=gpu:1
                    #SBATCH --ntasks=1
                    #SBATCH --nodes=1
                    #SBATCH --cpus-per-task=10

                    apptainer exec --nv miniconda3-user.sif python nonsequential_qnn.py

            .. tab-item:: nonsequential_qnn.py
                
                .. code-block:: python

                    import torch
                    import pennylane as qml
                    import numpy as np
                    from sklearn.datasets import make_moons


                    # Rastgele sayılar için tohum değerlerini belirleme 
                    torch.manual_seed(42)
                    np.random.seed(42)

                    X, y = make_moons(n_samples=200, noise=0.1)
                    y_ = torch.unsqueeze(torch.tensor(y), 1)  # one-hot encoding ile kodlanmış etiketler
                    y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

                    n_qubits = 2
                    dev = qml.device("lightning.gpu", wires=n_qubits)

                    # Quantum Node oluşturma
                    @qml.qnode(dev)
                    def qnode(inputs, weights):
                        qml.AngleEmbedding(inputs, wires=range(n_qubits))
                        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
                        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

                    n_layers = 6
                    weight_shapes = {"weights": (n_layers, n_qubits)}

                    # Quantum Node'u kuantum katmanına çevirme
                    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

                    # Modeli oluşturma
                    class HybridModel(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.clayer_1 = torch.nn.Linear(2, 4)
                            self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
                            self.qlayer_2 = qml.qnn.TorchLayer(qnode, weight_shapes)
                            self.clayer_2 = torch.nn.Linear(4, 2)
                            self.softmax = torch.nn.Softmax(dim=1)

                        def forward(self, x):
                            x = self.clayer_1(x)
                            x_1, x_2 = torch.split(x, 2, dim=1)
                            x_1 = self.qlayer_1(x_1)
                            x_2 = self.qlayer_2(x_2)
                            x = torch.cat([x_1, x_2], axis=1)
                            x = self.clayer_2(x)
                            return self.softmax(x)

                    model = HybridModel()

                    # Modeli eğitme
                    opt = torch.optim.SGD(model.parameters(), lr=0.2)
                    loss = torch.nn.L1Loss()

                    X = torch.tensor(X, requires_grad=True).float()
                    y_hot = y_hot.float()

                    batch_size = 5
                    batches = 200 // batch_size

                    data_loader = torch.utils.data.DataLoader(
                        list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
                    )

                    epochs = 6

                    for epoch in range(epochs):

                        running_loss = 0

                        for xs, ys in data_loader:
                            opt.zero_grad()

                            loss_evaluated = loss(model(xs), ys)
                            loss_evaluated.backward()

                            opt.step()

                            running_loss += loss_evaluated

                        avg_loss = running_loss / batches
                        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

                    y_pred = model(X)
                    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

                    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
                    accuracy = sum(correct) / len(correct)
                    print(f"Accuracy: {accuracy * 100}%")