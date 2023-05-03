Comparação de Locutor Multi Características Bayesiano
============

__autor:__ Adelino Pinheiro Silva

__email:__ adelinocpp@yahoo.com, adelinocpp@gmail.com, adelino@ufmg.br

__tel:__ +55 31 98801-3605

## Sistema operacional

Desenvolvido em: Linux Mint 20.3; Linux 5.4.0-126-generic; x86-64

Pré-requisitos: MediaInfoLib - v19.09

### Sequenciamento de treinamento e uso

1 - P01_Computa_caracteristicas_pragmaticas.py
2 - P02_Verifica_calculo_caracteristicas.py
3 - P03_Constroi_conjuntos_de_locutores_v0.py
P03_Constroi_UBM_v0.py
P04_Treina_TDDNN_v0.py
P05_Treina_RESNET_v0.py
P06_Treina_GMM_v0.py
P07_Computa_TDDNN_xVector_Calibrate_Validation_LDA_v0.py
P07_Compute_T_Matrix_by_BW_Statistics_v0_DEPRECATE.py
P07_Treina_BW_Statistics_and_T_Matrix_v0_DEPRECATE.py
P08_Computa_RESNET_xVector_Calibrate_Validation_LDA_v0.py
P08_Computa_TDDNN_xVector_Calibrate_Validation_LDA_v0.py
P09_Computa_RESNET_xVector_Calibrate_Validation_LDA_v0.py
P09_Treina_BW_Statistics_and_T_Matrix_v0.py
P10_Computa_iVector_Calibrate_Validation_LDA_v0.py
P11_Compute_Calibrate_Validate_GMM_UBM_BSW_WSV_v0_DEPRECATED.py
P12_Train_LDA_Sphering_models_v0.py
P13_Compute_Calibrate_Validate_GMM_UBM_BSW_WSV_v0.py
P14_Train_SPH_LDA_PLDA_Calibrate_Validate_ixvectors_v0.py
P15_Analise_Calibrate_Validate_ixvectors_v0.py
P16_Plotagem_Analises_v0.py

### Estrutura de diretórios

Criar um diretório principal "main_folder" e nele adicionar dois sub-diretŕorios. O primeiro "BaseDadosAudios" onde serão armazenados os arquivos de áudio para o processo. 

O segundo "Rotinas" onde __será depositado o conteúdo deste repositório__. A medida que as rotinas forem sendo executadas novos diretóris serão criados.

Exemplo da estrutura de diretórios inicial.
```
main_folder
     |-BaseDadosAudios
     |-Rotinas
     	|-P01_Computa_caracteristicas_pragmaticas.py
     	|-P02_Constroi_conjuntos_de_locutores_v0.py
        ...
```

#### Diretório "BaseDadosAudios"

O diretório de arquivos de áudio foi panejado para armazenar cada corpus (banco de dados de arquivos de áudio) em um subdiretório. O nome do subdiretório não é relevante. Em cada corpus é utilizado um diretŕorio para cada falante diferente. O material de cada falante foi dividido em dois arquivos "enroll.wav" e "test.wav".

É importante que cada diretório tenha um nome único. Para isso utilizei oito dígitos, os quatro primeiros referentes oa número do corpus e os quatro últimos ao número do locutor. Por exemplo, o diretŕorio 00320101 é o locutor de índice 101 do corpus de numero 32.

É importante que um mesmo locutor (indivíduo) nao esteja presente em mais de um corpus.

Estou providenciando uma forma de disponibilizar as bases de dados já condicionadas.

Exemplo da estrutura da base de dados de áudios.
```
BaseDadosAudios
	|-Diretorio_Corpus_01
    	|-00010001
        	|-enroll.wav
            |-test.wav
        |-00010002
        	|-enroll.wav
            |-test.wav
	|-Diretorio_Corpus_02
    	|-00020001
        	|-enroll.wav
            |-test.wav
        |-00020002
        	|-enroll.wav
            |-test.wav
	...
```



## Opções de execução

P01_Computa_caracteristicas_pragmaticas.py

Opções:

```
CONVERT_AUDIO_TO_PROCESS = True
COMPUTE_FEATURES = True
MULTI_CORES = True
```



