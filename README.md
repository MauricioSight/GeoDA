# Geometric Decision-based Attack (GeoDA)

Bem-vindo ao repositório do algoritmo GeoDa, uma ferramenta avançada para condução de ataques adversariais baseados em decisões geométricas. Este projeto visa reproduzir e aprimorar o GeoDa, explorando sua eficácia em diferentes normas e cenários.

## Visão Geral da Performance do GeoDA

Confira abaixo um exemplo visual da performance do GeoDa aplicado a diferentes normas, demonstrando sua capacidade de manipular efetivamente imagens para testar a robustez de modelos de classificação.

![Demo](https://user-images.githubusercontent.com/36679506/75689719-aa821b00-5c6f-11ea-9b6b-b78ff3ed871b.jpg)

## Estrutura do Repositório

### GeoDA.py

Este arquivo contém a implementação original do algoritmo GeoDa, com ajustes pontuais para correção de bugs, mantendo-se fiel ao design proposto pelos autores.

### GeoDA_resolved.ipynb

Um Jupyter Notebook que oferece uma reprodução detalhada do algoritmo GeoDa, incorporando melhorias e a realização de experimentos adicionais para explorar sua capacidade e eficácia.

### log.csv

Um registro detalhado de todos os experimentos conduzidos durante este projeto, permitindo uma análise aprofundada dos resultados e das performances observadas.

### Resnet50Flowers102/

Contém o modelo treinado utilizando a arquitetura ResNet50, especializado no dataset Flowers102. Este modelo serve como um dos principais objetos de teste para o algoritmo GeoDa em nossos experimentos.

### utils.py

Um conjunto de funções auxiliares utilizadas ao longo dos experimentos, facilitando operações comuns e permitindo uma reprodução mais eficiente e organizada do algoritmo.

### data/

Este diretório inclui algumas das imagens utilizadas nos experimentos, exemplificando os tipos de inputs que foram manipulados pelo GeoDa.

## Referências

Para uma compreensão mais profunda do GeoDa, recomendamos a consulta ao trabalho original dos autores:

[1] Ali Rahmati, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard, e Huaiyu Dai, *GeoDA: a geometric framework for black-box adversarial attacks*. Apresentado na conferência CVF/IEEE Computer Vision and Pattern Recognition (CVPR'20), 2020. [[arXiv pre-print]](http://arxiv.org/abs/2003.06468)

Este repositório é um esforço para explorar, validar e expandir a aplicabilidade do GeoDa, contribuindo para o campo de segurança em IA com uma ferramenta poderosa para testar a robustez de modelos contra ataques adversários.
