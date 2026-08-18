# Rede Neural Fashion MNIST — Do Zero

Uma rede neural totalmente conectada treinada no conjunto de dados [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), implementada do zero usando apenas NumPy. Sem PyTorch. Sem TensorFlow. Cada etapa de propagação direta, ativação e retropropagação é escrita manualmente.

---

## Visão Geral

Este projeto constrói um perceptron multicamadas (MLP) do zero para classificar 10 categorias de roupas e acessórios. O objetivo foi desenvolver uma intuição profunda de como redes neurais realmente funcionam — gradientes, atualizações de pesos e tudo mais — em vez de depender de frameworks com autograd para abstrair isso.

**Conjunto de dados:** Fashion MNIST — 70.000 imagens em escala de cinza 28×28 em 10 classes  
**Framework:** NumPy puro  
**Acurácia:** 70~75% no conjunto de teste

---

## O que foi Implementado

- **Propagação direta**: camadas lineares com multiplicação de matrizes manual
- **Funções de ativação**: ReLU (camadas ocultas), Softmax (camada de saída)
- **Função de perda**: perda de entropia cruzada
- **Retropropagação**: cálculo completo de gradientes manualmente
- **Atualização de parâmetros**: descida do gradiente com taxa de aprendizado configurável
- **Pré-processamento de dados**: normalização e codificação one-hot

---

## Arquitetura

```
Input (784)  →  Dense (128, ReLU)  →  Dense (64, ReLU)  →  Output (10, Softmax)
```

---

## Primeiros Passos

**Requisitos**
```
numpy
pandas
matplotlib
```

Instale com:
```bash
pip install numpy pandas matplotlib
```

**Execução**

Abra `Fashion_MNIST_classifier_from_scratch.ipynb` no Jupyter e execute todas as células. O conjunto de dados é carregado diretamente do formato CSV do Kaggle — faça o download [aqui](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

---

## Classes

| Rótulo | Classe |
|-------|-------|
| 0 | Camiseta/top |
| 1 | Calça |
| 2 | Suéter |
| 3 | Vestido |
| 4 | Casaco |
| 5 | Sandália |
| 6 | Camisa |
| 7 | Tênis |
| 8 | Bolsa |
| 9 | Bota de cano curto |

---

## Referências

- [Neural Networks from Scratch — Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [Simple MNIST NN from Scratch (NumPy)](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras)
- [Fashion MNIST Dataset — Zalando Research](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- [Fashion MNIST Paper — Xiao et al., 2017](https://arxiv.org/pdf/1802.01528)
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/index.html)
- [An Introduction to Deep Learning — Guilhoto](https://math.uchicago.edu/~may/REU2018/REUPapers/Guilhoto.pdf)
