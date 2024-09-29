import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Define the poker hands and their scores
POKER_HAND_SCORES = {
    'no_hand': 0,
    'pair': 5,
    'two_pairs': 60,
    'flush': 80,
    'three_of_a_kind': 125,
    'straight': 180,
    'four_of_a_kind': 325,
    'straight_flush': 450,
}

# Define the deck
DECK = [f'{rank}{suit}' for rank in '6789TJQKA' for suit in 'SHDC']  # 36-card deck

class PileupPokerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(PileupPokerEnv, self).__init__()
        
        self.grid_size = 4
        self.num_cards_per_round = 5
        self.rounds = 4
        self.deck = DECK.copy()
        random.shuffle(self.deck)
        # Uncomment to use a specific deck for training
        # I had this to see if we can overfit over a deck
        # self.use_this_deck()

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_cards_per_round * (self.grid_size * self.grid_size))
        self.observation_space = spaces.Dict({"grid": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, len(DECK)), dtype=np.float32),
                                              "hand": spaces.Box(low=0, high=1, shape=(self.num_cards_per_round, len(DECK)), dtype=np.float32)})

        # Initialize the game state
        self.grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_round = 0
        self.score = 0
        self.multiplier = 1
        self.num_moves_made = 0
        self.selected_card_indices = []
        self.used_cards = []
        self.num_total_moves = 0
        self.render_mode = render_mode
        self.discard_hand = []

        # Used only for render
        self.hand = [self.deck[i] for i in range(self.num_cards_per_round)]
        self.current_choice = (None, None)
        self.actions_made_this_game = []
        self.hand_score_array = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.deck = DECK.copy()
        random.shuffle(self.deck)
        # Uncomment to use a specific deck for training
        # I had this to see if we can overfit over a deck
        # self.use_this_deck()
        self.grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_round = 0
        self.score = 0
        self.multiplier = 1
        self.num_moves_made = 0
        self.selected_card_indices = []
        self.used_cards = []
        self.num_total_moves = 0
        self.hand = [self.deck[i] for i in range(self.num_cards_per_round)]
        self.current_choice = (None, None)
        self.actions_made_this_game = []
        self.hand_score_array = []
        self.discard_hand = []
        return self._get_observation(), {}

    def step(self, action):
	    # Decode the flattened action
        card_index = action // (self.grid_size * self.grid_size)
        pos = action % (self.grid_size * self.grid_size)
        # Failsafe, should not enter this if condition
        if self.current_round >= self.rounds:
            return self._get_observation(), self.score, True, {}, {}
        # Decode the position and card
        row, col = divmod(pos, self.grid_size)
        card = self.deck[card_index]
        self.current_choice = (card, (int(row), int(col)))
        # Append it to action list
        self.actions_made_this_game.append(self.current_choice)
        
        # Valid move
        if self.grid[row][col] == '' and (card not in self.used_cards):
            self.used_cards.append(card)
            # Place card on the grid
            self.grid[row][col] = card
            self.num_moves_made += 1
            # End of round
            if (self.num_moves_made == 4):  
                self.current_round += 1
                # Pop all cards
                for i in range(self.num_cards_per_round):
                    # Check for the discard card and put it in discard hand
                    if(self.deck[0] not in self.used_cards):
                        self.discard_hand.append(self.deck[0])
                    self.deck.pop(0)
                # Reset necessary parameters
                self.num_moves_made = 0
                self.hand = [self.deck[i] for i in range(self.num_cards_per_round)]
                # Game over state
                if self.current_round >= self.rounds:
                    self.score = 0.0
                    self.score += self._calculate_score()
                    return self._get_observation(), self.score, True, {}, {}
            # Game not done yet, but round is done
            # Small positive reward for making a good move
            self.score = 1.0
            # Add the round scores
            # Not sure if we want to add it this way. While this makes sure
            # rewards are not sparse, it may also encourage greedy behavior?
            self.score += self._calculate_score()
            return self._get_observation(), self.score, False, {}, {}
        # Bad move
        return self._get_observation(), -1, False, {"invalid_action": True}, {}
    
    # Some helpers for greedy agent
    # Can be used for other agents too
    def get_empty_positions_in_grid(self):
        empty_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == '']
        return empty_positions
    
    def get_grid_copy(self):
        return np.copy(self.grid)
    
    def place_card_on_grid(self, card, row, col):
        self.grid[row][col] = card

    def erase_card_on_grid(self, row, col):
        self.grid[row][col] = ''
    
    # By pass function to obtain score for a round
    def get_total_score(self):
        out_score = self._calculate_score()
        if(out_score is None):
            out_score = 0
        return out_score
    
    # Score calculator
    def _calculate_score(self):
        def is_flush(cards):
            return len(set(suit for rank, suit in cards)) == 1

        def is_straight(cards):
            ranks = sorted(['6789TJQKA'.index(rank) for rank, suit in cards])
            return ranks == list(range(ranks[0], ranks[0] + 4))

        def get_hand_score(cards):
            ranks = [rank for rank, suit in cards]
            unique_ranks = set(ranks)
            rank_counts = {rank: ranks.count(rank) for rank in unique_ranks}

            # Four of a Kind or Full House
            if len(unique_ranks) == 1:  
                return POKER_HAND_SCORES['four_of_a_kind']
            # Three of a Kind or Two Pairs
            elif len(unique_ranks) == 2:
                if 3 in rank_counts.values():
                    return POKER_HAND_SCORES['three_of_a_kind']
                else:
                    return POKER_HAND_SCORES['two_pairs']
            # Pair
            elif len(unique_ranks) == 3:  
                return POKER_HAND_SCORES['pair']
            elif is_straight(cards) and is_flush(cards):
                return POKER_HAND_SCORES['straight_flush']
            elif is_flush(cards):
                return POKER_HAND_SCORES['flush']
            elif is_straight(cards):
                return POKER_HAND_SCORES['straight']
            else:
                return POKER_HAND_SCORES['no_hand']
            
        # Called when a row/column/corner hand is not full
        # Checks to see if a pair or a three of a kind is available
        def get_hand_score_for_p_or_3p(cards):
            ranks = [rank for rank, suit in cards]
            unique_ranks = set(ranks)

            if(len(cards) == 3):
                if(len(unique_ranks) == 1):
                    return POKER_HAND_SCORES['three_of_a_kind']
                elif(len(unique_ranks) == 2):
                    return POKER_HAND_SCORES['pair']
                else:
                    return POKER_HAND_SCORES['no_hand']
            elif(len(cards) == 2):
                if(len(unique_ranks) == 1):
                    return POKER_HAND_SCORES['pair']
                else:
                    return POKER_HAND_SCORES['no_hand']

        hands = []
        
        # Check row hands
        for i in range(self.grid_size):
            row_hand = [self.grid[i][j] for j in range(self.grid_size) if self.grid[i][j] != '']
            if len(row_hand) == self.grid_size:
                hands.append(get_hand_score(row_hand))
            elif (len(row_hand) > 1 and len(row_hand) < self.grid_size):
                hands.append(get_hand_score_for_p_or_3p(row_hand))
            else:
                hands.append(0)
        
        # Check column hands
        for j in range(self.grid_size):
            column_hand = [self.grid[i][j] for i in range(self.grid_size) if self.grid[i][j] != '']
            if len(column_hand) == self.grid_size:
                hands.append(get_hand_score(column_hand))
            elif (len(column_hand) > 1 and len(column_hand) < self.grid_size):
                hands.append(get_hand_score_for_p_or_3p(column_hand))
            else:
                hands.append(0)
        
        # Check corner hand
        corner_hand = [
            self.grid[0][0], self.grid[0][self.grid_size-1],
            self.grid[self.grid_size-1][0], self.grid[self.grid_size-1][self.grid_size-1]
        ]
        if all(corner_hand):
            # Corner hand is worth ×2
            hands.append(get_hand_score(corner_hand) * 2) 
        else:
            hands.append(0)

        num_hands = sum(x!=0 for x in hands)
        if(num_hands == 9):
            # Discard hand if counted is x3
            hands.append(get_hand_score(self.discard_hand) * 3) 
        else:
            hands.append(0)

        self.hand_score_array = hands

        # Calculate multiplier based on the number of valid hands
        num_hands = sum(x!=0 for x in hands)
        if num_hands >= 2 and num_hands <= 3:
            multiplier = 2
        elif num_hands >= 4 and num_hands <= 5:
            multiplier = 3
        elif num_hands >= 6 and num_hands <= 7:
            multiplier = 4
        elif num_hands >= 8 and num_hands <= 9:
            multiplier = 5
        elif num_hands == 10:
            multiplier = 6
        else:
            multiplier = 1

        self.multiplier = multiplier
        total_score = sum(hands) * self.multiplier
        return total_score


    def _get_observation(self):
        # Convert the grid into a one-hot encoded vector
        observation = np.zeros((self.grid_size, self.grid_size, len(DECK)), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] != '':
                    card_index = DECK.index(self.grid[i][j])
                    observation[i][j][card_index] = 1.0

        # Convert the current hand into a one-hot encoded vector
        hand_observation = np.zeros((self.num_cards_per_round, len(DECK)), dtype=np.float32)
        for idx, card in enumerate(self.hand):
            card_index = DECK.index(card)
            hand_observation[idx][card_index] = 1.0 
        
        return {"grid":observation,
                "hand":hand_observation}

    def render(self):
        print("----------------BEGIN RENDER----------------")
        print("-" * 25)
        for row in self.grid:
            print(" | ".join([card if card else "  " for card in row]))
        if self.current_round < self.rounds:
            print("Current Hand: " + " | ".join([card if card else "  " for card in self.hand]))
        print(f"Score: {self.score}, Multiplier: {self.multiplier}")
        print("Score array (row, col, corner): ", self.hand_score_array)
        print("Discard hand: ", self.discard_hand)
        print("----------------END RENDER----------------")

    def get_action_mask(self):
        # Initialize mask to True for all actions
        mask = [True for x in range(self.num_cards_per_round * self.grid_size * self.grid_size)]
        
        for card_index in range(self.num_cards_per_round):
            local_card = self.deck[card_index]
            for pos in range(self.grid_size * self.grid_size):
                row, col = divmod(pos, self.grid_size)
                if self.grid[row][col] != '' or (local_card in self.used_cards):
                    # Mark the action as invalid
                    mask[card_index * (self.grid_size * self.grid_size) + pos] = False

        return mask
    
    def set_deck(self, input_deck):
        '''
        set_deck() sets the input deck to the choice of the user.
        with set_deck() user can give custom decks and see 
        how their agent is performing

        input_deck argument should have 20 elements/cards and passed
        in the form of a list. Th deck is curated such that the first 5
        elements in input_deck form the first draw; next 5 the next draw and so on.
        '''

        if(len(input_deck) != 20):
            raise ValueError(f"Input deck {input_deck} does not have 20 elements. Unable to set deck")

        # Set the deck
        deck_copy = DECK.copy()
        self.deck = []
        for elem in input_deck:
            if elem in deck_copy:
                self.deck.append(elem)
                deck_copy.remove(elem)
            else:
                raise ValueError(f"Selected element {elem} is not in the original array or already used.")
        self.deck.extend(deck_copy)

        # Set current hand
        self.hand = [self.deck[i] for i in range(self.num_cards_per_round)]
    
    def set_deck_with_observation(self, input_deck):
        '''
        set_deck_with_observation() sets a deck and returns the observation

        We need to return the observation just like reset() in order to get the
        agents going with newly set observations.

        Always call after reset() for other parameters to be in place
        '''
        self.set_deck(input_deck)
        return self._get_observation(), {}
    
    def use_this_deck(self):
        '''
        use_this_deck() sets a custom deck defined inside of the function.
        I know its a little dirty, but I want to get the overfit process going
        without changing the original layout much
        '''
        ### Decks:
        deck_1 = ['JD', 'KD', 'QD', 'AD', '6S',
                'JH', 'AH', 'KH', 'QH', '6D',
                'KC', 'AC', 'QC', 'JC', '6H',
                'AS', 'JS', 'KS', 'QS', '6C']

        deck_2 = ['9D', '9C', '6C', '7D', 'KS',
                '7C', '6S', '9H', 'KD', '6H',
                '9S', 'QD', 'QC', 'AS', 'TH',
                'KH', 'KC', '7S', '8C', 'AD']

        deck_3 = ['JS', 'TH', 'QH', 'AH', 'AD',
                'KH', '6S', 'AC', '8H', '9C',
                '9D', '6D', 'JC', 'KC', '9H',
                'TD', '6C', '7H', '7D', '7S']

        deck_4 = ['QC', '9H', 'JH', '9S', '6S',
                '7C', '6H', 'QD', 'JD', 'KS',
                'KD', 'TC', '8D', '7S', '8C',
                'QS', 'TH', 'AS', 'KC', 'KH']
        
        self.set_deck(deck_1)

    def close(self):
        pass

