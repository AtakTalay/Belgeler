.. _qcomp_setup:


===============================
Qiskit ve PennyLane Kurulumları
===============================

--------------------------
Konteyner Oluşturma
--------------------------
İlgili paketleri kurmak için :ref:`Python ve Konteyner <python-container>` belgesinde de anlatıldığı gibi içine kurmak istediğiniz klasöre geldikten sonra konteyneri aşağıdaki gibi oluşturunuz.

.. code-block:: bash
    
    cp -r /arf/sw/containers/miniconda3/miniconda3-container.sif ./
    apptainer build --sandbox miniconda3-user miniconda3-container.sif

daha sonra konteyneri aşağıdaki gibi açınız.

.. code-block:: bash

    apptainer shell --no-home --writable --fakeroot miniconda3-user
    # veya
    apptainer shell --no-home --writable miniconda3-user


--------------------------
Qiskit Kurulumu
--------------------------

Qiskit'i ve onu GPU'da çalıştırmak için gerekli olan Aer-GPU kütüphanelerini aşağıdaki şekilde kurabilirsiniz.

.. code-block:: bash

    pip install qiskit==1.4.2
    pip install qiskit-aer-gpu

.. note::
    ``Qiskit machine learning`` gibi ekstra Qiskit kütüphanelerini kurmak için `Qiskit'in Dokümantasyonuna <https://qiskit.org/documentation/>`_ bakabilirsiniz.


----------------------------------------------
PennyLane Kurulumu
----------------------------------------------

Benzer şekilde Pennylane'i ve onu GPU'da çalıştırmak için gerekli olan Lightning-GPU kütüphanelerini aşağıdaki şekilde kurabilirsiniz.

.. code-block:: bash

    pip install pennylane
    pip install pennylane-lightning-gpu


Kurulum ve kullanım hakkında daha fazla bilgi için dokümantasyon `dokümantasyon <https://pennylane.ai/>`_ sayfasına bakabilirsiniz.



--------------------------
Kurulumları Test Etme
--------------------------
Kütüphanelerin kurulumunu tamamladıktan sonra konteynerden çıkmak için:

.. code-block:: bash

    exit

Sonrasında hazırlanan konteynerin imajını oluşturmak için:

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


Konteyner imajını oluşturduktan sonra Qiskit ve Pennylane kurulumlarını test etmek için aşağıdaki örnek kodları kullanabilirsiniz.

.. dropdown:: :octicon:`codespaces;1.5em;secondary` Qiskit GPU Testi(Tıklayınız)
    :color: info

        .. tab-set::

            .. tab-item:: İş Gönderme

                .. code-block:: bash

                    sbatch qiskit_test.slurm

            .. tab-item:: qiskit_test.slurm

                .. code-block:: bash
            
                    #!/bin/bash

                    #SBATCH --output=slurm-%j.out
                    #SBATCH --error=slurm-%j.err
                    #SBATCH --time=00:15:00
                    #SBATCH --job-name=qiskit_test

                    #SBATCH -p debug
                    #SBATCH --partition=akya-cuda
                    #SBATCH --gres=gpu:1
                    #SBATCH --ntasks=1
                    #SBATCH --nodes=1
                    #SBATCH --cpus-per-task=10

                    apptainer exec --nv miniconda3-user.sif python qiskit_test.py

            .. tab-item:: qiskit_test.py
                
                ..  code-block:: python

                    from qiskit import QuantumCircuit, transpile
                    from qiskit_aer import AerSimulator

                    # Kuantum devresini oluşturma
                    circ = QuantumCircuit(2)
                    circ.h(0)
                    circ.cx(0, 1)
                    circ.measure_all()                      

                    # GPU'da çalışacak simülatorü ayarlama
                    simulator = AerSimulator(device='GPU')
                    circ = transpile(circ, simulator)

                    # Simulasyonu çalıştırma
                    result = simulator.run(circ, shots=1024).result()
                    counts = result.get_counts(circ)

                    print(counts)


.. dropdown:: :octicon:`codespaces;1.5em;secondary` Pennylane GPU Testi(Tıklayınız)
    :color: info

        .. tab-set::

            .. tab-item:: İş Gönderme

                .. code-block:: bash

                    sbatch pl_test.slurm

            .. tab-item:: pl_test.slurm

                .. code-block:: bash
            
                    #!/bin/bash

                    #SBATCH --output=slurm-%j.out
                    #SBATCH --error=slurm-%j.err
                    #SBATCH --time=00:15:00
                    #SBATCH --job-name=pl_test

                    #SBATCH -p debug
                    #SBATCH --partition=akya-cuda
                    #SBATCH --gres=gpu:1
                    #SBATCH --ntasks=1
                    #SBATCH --nodes=1
                    #SBATCH --cpus-per-task=10

                    apptainer exec --nv miniconda3-user.sif python pl_test.py

            .. tab-item:: pl_test.py
                
                ..  code-block:: python

                    import pennylane as qml

                    # GPU'da çalışacak simülatorü ayarlama
                    dev = qml.device('lightning.gpu', wires=2, shots=1024)

                    @qml.qnode(dev)
                    def circuit():
                        # Kuantum devresini oluşturma
                        qml.Hadamard(0)
                        qml.CNOT(wires=[0, 1])

                        return qml.counts()

                    # Simulasyonu çalıştırma
                    counts = circuit()
                    print(counts)