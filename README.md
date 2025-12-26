# Optimization and Dynamics of Learning

This project explores how the choice of optimization method influences the convergence and generalization of deep models.  

Datasets: MNIST, CIFAR-10  
Optimizers: SGD, Momentum, RMSProp, Adam  
Learning rate schedules, sharp vs. flat minima  
    
## Abstract

Optimization of deep neural networks is essential for efficient training, but the dynamics of various optimization algorithms and strategies are not fully understood. Observing how different optimizers affect
convergence and generalization can guide better algorithm choices. In this work, we study the optimization dynamics of convolutional neural networks on two well-known datasets : MNIST and CIFAR10. We compare different optimization algorithms such as SGD, RMSProp, Adagrad and Adam under varying learning rates and batch sizes, analyzing loss landscapes and convergence behavior. 
