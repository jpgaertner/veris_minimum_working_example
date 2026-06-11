# Veris minimum working example

Scripts to run Veris as a standalone model with benchmark forcing fields. The dynamics and thermodynamics components are separated here but can easily be combined.

This branch uses the redesigned version of Veris ([jax-only branch](https://github.com/jpgaertner/veris/tree/jax-only)), restructured around JAX's sharded arrays.

To use these scripts, you need to install Veris either via pip (```pip install veris```) or from the [Veris Github repository](https://github.com/jpgaertner/veris/tree/jax-only)) (```pip install -e .```) if you want to modify the model code.
