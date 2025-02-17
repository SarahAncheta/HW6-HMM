import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables

        T = len(input_observation_states)
        N = self.hidden_states.shape[0]

        fwrd_mat = np.zeros((N, T))

        for s in range(N):
            state_initial_index = self.observation_states_dict[input_observation_states[0]]

            fwrd_mat[s, 0] = self.prior_p[s] * self.emission_p[s, state_initial_index]
        
       
        # Step 2. Calculate probabilities

        for t in range(1, T):

            my_state_index = self.observation_states_dict[input_observation_states[t]]

            for s in range(N):
                # got this clean summation idea across the rows of the matrix from ChatGPT
                fwrd_mat[s, t] = np.sum(fwrd_mat[:, t-1]* self.transition_p[:, s] * self.emission_p[s, my_state_index])
                 
        forwardprob = np.sum(fwrd_mat[:,-1])

        # Step 3. Return final probability 

        return forwardprob
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states), dtype=int)
        #I believe this should be the same length 
    
        N = self.hidden_states.shape[0]
        T = len(decode_observation_states)

         #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((N, T))

        back_pointer = np.zeros((N, T), dtype=int)

       
       # Step 2. Calculate Probabilities
        for s in range(N):
                state_initial_index = self.observation_states_dict[decode_observation_states[0]]
                viterbi_table[s, 0] = self.prior_p[s] * self.emission_p[s, state_initial_index]
                back_pointer[s, 0] = 0
        for t in range(1, T):
                my_state_index = self.observation_states_dict[decode_observation_states[t]]
                for s in range(N):
                     possible_values = viterbi_table[:, t-1] * self.transition_p[:, s]
                     viterbi_table[s, t] = np.max(possible_values)*self.emission_p[s, my_state_index]
                     back_pointer[s, t] = np.argmax(possible_values)
        
        # Step 3. Traceback 

        best_path_prob = np.max(viterbi_table[:,-1])
        best_path_pointer = np.argmax(viterbi_table[:,-1])

        # TO DO: create bestpath, starting at state best_path_pointer, 
        # follows back_pointer to states back in time

        best_path[-1] = best_path_pointer

        #asked chatgpt for help in figuring out indexing in this line
        for state in range(T-2, -1, -1):
             best_path[state] = back_pointer[best_path[state+1], state+1]
             
        # Step 4. Return best hidden state sequence 

        best_hidden_state_sequence = [str(self.hidden_states[i]) for i in best_path]

        return best_hidden_state_sequence, best_path_prob
        