Intra-tumoral stroma percentage computer
========================================

.. image:: https://github.com/NKI-AI/itsper/blob/main/assets/ITSP.png
   :alt: Illustration of outputs obtained on a TCGA breast resection using itsper

itsper is a simple command line utility which can be used to quantify the ITSP biomarker from tissue segmentations of histopathological tumor lesions. The library assumes that the tissue segmentations were generated using models from the ahcore library.
It was developed at the `AI for Oncology lab <https://aiforoncology.nl>`_.

How to install
==============
1. clone this repository
2. cd <repo>
3. pip install -e .

Usage
=====
Type `itsper -h` to know more.

License
-------

itsper is not intended for clinical use. It is licensed under the `MIT License <https://mit-license.org/>`_.
