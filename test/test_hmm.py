import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p'] 
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']
    observation_state_sequence = mini_input['observation_state_sequence']
    best_hidden_state_sequence = mini_input['best_hidden_state_sequence']
   
    myHMM = HiddenMarkovModel(observation_states=observation_states, 
                              hidden_states=hidden_states, 
                              prior_p=prior_p,
                              transition_p=transition_p, 
                              emission_p=emission_p)
    
    forward_prob = myHMM.forward(observation_state_sequence)

    assert np.isclose(forward_prob, 0.03506441162109375, rtol=1e-05)

    predict_states, best_prob_path = myHMM.viterbi(observation_state_sequence)

    assert np.isclose(best_prob_path, 0.0066203212359375, rtol=1e-05)

    for i in range(len(best_hidden_state_sequence)):
        assert predict_states[i] == best_hidden_state_sequence[i]

    assert len(predict_states) == len(best_hidden_state_sequence)



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')
    observation_states = full_hmm['observation_states']
    hidden_states = full_hmm['hidden_states']
    prior_p = full_hmm['prior_p'] 
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']
    observation_state_sequence = full_input['observation_state_sequence']
    best_hidden_state_sequence = full_input['best_hidden_state_sequence']

    my_full_HMM = HiddenMarkovModel(observation_states=observation_states, 
                              hidden_states=hidden_states, 
                              prior_p=prior_p,
                              transition_p=transition_p, 
                              emission_p=emission_p)
    
    forward_prob = my_full_HMM.forward(observation_state_sequence)

    assert np.isclose(forward_prob, 1.6864513843961343e-11, rtol=1e-11)

    predict_states, best_prob_path = my_full_HMM.viterbi(observation_state_sequence)

    assert np.isclose(best_prob_path, 2.5713241344000005e-15, rtol=1e-15)

    for i in range(len(best_hidden_state_sequence)):
        assert predict_states[i] == best_hidden_state_sequence[i]

    assert len(predict_states) == len(best_hidden_state_sequence)















