[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/XLECW-4r)
# Ativiadade Ciência da Computação
<img src="assets/images/Unicap_Icam_Tech-01.png" alt="drawing" width="200"/>

## Identificação
**Professor**: Diego Pinheiro, PhD

**Disciplina**: Ciência dos Dados

**Atividade**: Atividade 5 - Classificação com Regressão Logística (Perceptron Learning Algorithm)

## Instruções 
> 1. Sua implementação deve estar dentro da pasta **src** 
> 2. Não modifique nenhum código dentro da pasta **test**
> 3. A submissão **não deve ser feita após o prazo** (nem 1 minuto a mais)

## Descrição da Atividade
### Questão 01

Implemente a estratégia `Steepest Descent` que atualiza o peso conforme a equação a seguir:
$$
\mathbf{w}^{[t+1]} = \mathbf{w}^{[t]} - \lambda \nabla E(\mathbf{w})
$$


Implemente a estratégia `Newton’s Method`  que atualiza o peso conforme a equação a seguir:

$$
\mathbf{w}^{[t+1]} = \mathbf{w}^{[t]} - \lambda H^{-1}\nabla E(\mathbf{w})
$$


O log-likelihood é 
$$
l(\mathbf{w}) = \sum_{i=1}^{N}\{y_i log(p_i) + (1 - y_i)(1 - log(p_i))\}
\\
= \sum_{i=1}^{N} \{{y_i}\mathbf{w}^T\mathbf{x_i} - log(1 + e^{\mathbf{w}^T\mathbf{x_i}}) \}
$$

Portanto, o erro é 
$$
E(\mathbf{w}) = -l(\mathbf{w})
$$

A derivada primeira do erro (ie., gradiente) é 
$$
\frac{\partial E(\mathbf{w})}{\partial \mathbf{w}} = \nabla E(\mathbf{w}) 
\\
= \sum_{i=1}^{N} \{\mathbf{x_i}(y_i - p_i)\}
\\
= X^T(\mathbf{p} - \mathbf{y})
$$

A derivada segunda do error (ie., Hessiana) é 

$$
\frac{\partial^2 E(\mathbf{w})}{\partial \mathbf{w} \partial \mathbf{w}^T} = \nabla \nabla E(\mathbf{w})
\\
= \sum_{i=1}^{N} \mathbf{x_i} p_n (1 - p_n)\mathbf{x_i}^T 
\\
= X^TRX
$$


em que $P$ é a matriz diagonal 
$$
P_{nn} = p_n(1 - p_n)
$$


