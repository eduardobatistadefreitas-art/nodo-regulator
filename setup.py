from setuptools import setup

setup(
    name="nodo-regulator",
    version="0.1.0",
    author="Engenharia de Dados",
    description="Filtro espectral de alta performance para redução de dimensionalidade O(N)",
    py_modules=["nodo_regulator"],  # Mapeia diretamente o arquivo único na raiz
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
