# cv-foundations-6

## Conteúdo
 1. [Requisitos](#requisitos)
 2. [Estrutura](#estrutura)
 3. [Uso](#uso)

## Requisitos 
1.  Python 3.5.2	
2.  OpenCV 3.3.0
3.  Keras 2.2.4
4.  Matplotlib 2
5.  Scikit-learn 0.20
6.  Tensorflow 1.11

## Estrutura
- Pasta relatorio com código fonte do relatório
- Arquivo Araujo_Pedro__Ramos_Raphael.pdf com o relatório
- Pasta src contendo o código principal do projeto

## Uso
- [Repositório do github](https://github.com/peluz/cv-foundations-6)
- Requisito 1:
	- rodar no diretório raiz do projeto, para gerar o dicionário de imagens:
	```bash
	python src/build_data.py
	```
	- em seguida rodar, para treinar o modelo com diferentes hiperparâmetros e avaliá-los:
	```bash
	python src/train.py
	```
- Requisito 2:
