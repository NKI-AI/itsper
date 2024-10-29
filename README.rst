Intra-tumoral stroma percentage computer
========================================

.. image:: https://github.com/NKI-AI/itsper/blob/main/assets/ITSP.png
   :alt: Illustration of outputs obtained on a TCGA breast resection using itsper

itsper is a simple command line utility which can be used to compute the intra-tumoral stroma percentage (ITSP) biomarker from tissue segmentations of histopathological tumor lesions. It was developed at the `AI for Oncology lab <https://aiforoncology.nl>`_.

Note: Tissue segmentations have to be generated using models from the `ahcore <https://github.com/NKI-AI/ahcore>`_ library.


How to install
==============
To install itsper, run the following commands:

.. code-block:: bash

    git clone https:/github.com/NKI-AI/itsper/
    cd itsper
    pip install -e .

Usage
=====
Type `itsper -h` to know more.

License
=======
itsper is not intended for clinical use. It is licensed under the `MIT License <https://mit-license.org/>`_.

Citing ITSPER
=============

If you use ITSPER in your research, please use the following citation:

.. code-block:: bibtex

    @software{itsper,
      author = {Pai, A.},
      month = {10},
      title = {{ITSPER: Intra-Tumoral Stroma Percentage Computer}},
      url = {https://github.com/NKI-AI/itsper},
      version = {0.1.1},
      year = {2024}
    }
