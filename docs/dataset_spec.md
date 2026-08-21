# ChangeRisk — Especificação do Dataset

## Objetivo

Construir um dataset supervisionado para estimar o risco de uma Pull Request apresentar evidência posterior de defeito após sua integração ao projeto.

## Repositório

pandas-dev/pandas

## Unidade de análise

Cada observação representa uma Pull Request integrada ao repositório.

## Período das mudanças

Serão consideradas Pull Requests integradas entre:

- 01/01/2022
- 31/12/2025

## Momento da previsão

A previsão é realizada imediatamente antes do merge da Pull Request.

Somente informações disponíveis até esse momento poderão ser utilizadas como features.

Informações posteriores ao merge não poderão ser utilizadas como variáveis preditoras.

## Código de produção

Uma Pull Request será considerada elegível para o dataset de Machine Learning quando modificar pelo menos um arquivo localizado em:

pandas/

Serão excluídos desse critério arquivos localizados em:

pandas/tests/

PRs exclusivamente de documentação, testes, CI ou configuração poderão ser preservadas nos dados brutos, mas não serão utilizadas no dataset principal de treinamento.

## Janela de observação

O comportamento posterior à mudança será observado durante 90 dias após o merge.

Para garantir janela completa para as PRs de 2025, serão consideradas correções integradas até:

31/03/2026

## Variável-alvo

Nome:

observed_defect_90d

### Valor 1

A PR possui evidência de defeito posterior de alta confiança.

A evidência exige:

1. correção posterior dentro de 90 dias;
2. PR posterior classificada como Bug;
3. título da correção iniciado por BUG;
4. alteração em código de produção relacionado;
5. evidência SZZ indicando que código corrigido foi atribuído à PR original.

### Valor 0

Nenhuma evidência de defeito foi identificada pelo procedimento definido dentro da janela de 90 dias.

O valor 0 não significa garantia de ausência de defeito.

### Não aplicável

PRs que não alteram código de produção ou que não possuem janela completa de observação não recebem target válido para treinamento.

## Evidência SZZ

O procedimento utiliza:

bug fix
→ diff da correção
→ linhas removidas ou substituídas
→ git blame no estado anterior à correção
→ comparação com o commit da PR original

O resultado bruto do SZZ será preservado separadamente da classificação de alta confiança.

## Controle de data leakage

Features deverão ser calculadas exclusivamente com dados existentes até o momento do merge.

Nenhuma informação proveniente dos 90 dias posteriores poderá ser utilizada como feature.

O período posterior será utilizado exclusivamente para construção da variável-alvo.

## Estratégia temporal de avaliação

Treino:
2022–2023

Validação:
2024

Teste final:
2025

Essa separação simula o cenário real de utilizar mudanças históricas para prever o risco de mudanças futuras.