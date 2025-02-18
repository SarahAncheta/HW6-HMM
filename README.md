# HW6-HMM

Sarah Ancheta

In this assignment, I implemented the Forward and Viterbi Algorithms (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

This code is based off of the psudeocode found in [this textbook](https://web.stanford.edu/~jurafsky/slp3/A.pdf).

The forward algorithm computes the probability of a given observed sequence, calculating based on all possible hidden options.

The Viterbi algorithm computes the most likely sequence of hidden states that could have resulted in a given observed sequence, and also returns the probability of that most likely sequence.

I included code for the following two edge cases:
1. Check that the probability distributions add up to one, and if they do not, then we raise a value error asking to scale.
2. Check that the dimensions of the inputs for the hidden state information matches. 


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
