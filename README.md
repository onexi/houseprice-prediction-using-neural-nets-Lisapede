[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DUGMT0Yz)
# PS3NeuralNetHousePrice


Notes on how I approached the assignment:

----
THE LEARNING RATE

The original output gives me the following:
Test Mean Squared Error: 926390656.0

then I increased the learning rate from lr=0.001 to lr = 0.01
Test Mean Squared Error: 911584000.0
so it slighly improved...

let's try lr = 0.005
Test Mean Squared Error: 892385920.0
this rate is better.

-----
INCREASING THE NUMBER OF NEURONS
Test Mean Squared Error: 909465216.0
that increased the error
reran code > Test Mean Squared Error: 910590592.0

# Define the neural network model with regularization and batch normalization
Test Mean Squared Error: 955838208.0
this also increased the error...

adding more structure also increased the error...
Test Mean Squared Error: 974426944.0

--------
NEED TO CHANGE OUR APPROACH
revisited the order of the x and y train / test and reran code
Test Mean Squared Error: 0.1261841207742691

_________

