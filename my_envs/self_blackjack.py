import os
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from random import uniform


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are two actions: stick (0), and hit (1).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:

        +1.5 (if <a href="#nat">natural</a> is True)

        +1 (if <a href="#nat">natural</a> is False)

    ### Arguments

    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab">`sab=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (
                spaces.Tuple((spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))),
                spaces.Tuple((spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))),
            )
        )

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.render_mode = render_mode

        self.terminated = False
        self.reward = 0.0

        # --- NEW: sequential (turn-based) state for player & dealer ---
        # player_stick / dealer_stick: once True, that side can't take more cards
        # current_turn: "player" or "dealer", indicates whose action this step() applies to
        self.player_stick = False
        self.dealer_stick = False

        # Get a random number between 0 and 1
        sample = uniform(0, 1)

        # If the number is between [0.5,1.0], player starts
        if sample >= 0.5:
            self.current_turn = "player"

        # If the number is between [0.0,0.5], dealer starts
        elif sample < 0.5:
            self.current_turn = "dealer"

    def _get_obs(self):
        player_obs = (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
        dealer_obs = sum_hand(self.dealer), self.player[0], usable_ace(self.dealer)
        return (player_obs, dealer_obs)

    def _final_result(self):
        """
        Compute the final reward from the player's perspective after both
        player and dealer have stuck (and neither has just busted).
        """
        r = cmp(score(self.player), score(self.dealer))

        if self.sab and is_natural(self.player) and not is_natural(self.dealer):
            r = 1.0
        elif (not self.sab) and self.natural and is_natural(self.player) and r == 1.0:
            r = 1.5

        return float(r)

    def step(self, action):
        """
        Sequential (turn-based) variant:
        - One player and one dealer.
        - They take turns choosing hit(1) / stick(0) via env.step(action).
        - Once stick, that side can no longer hit.
        - Reward is always from the player's perspective.
        """
        assert self.action_space.contains(action)
        assert self.current_turn in ("player", "dealer")
        suits = ["C", "D", "H", "S"]

        # ----- Player's turn -----
        if self.current_turn == "player":
            if action:  # hit
                card = draw_card(self.np_random)
                self.player.append(card)
                suit = self.np_random.choice(suits)
                if card == 1:
                    card_str = "A"
                elif card == 10:
                    card_str = self.np_random.choice(["J", "Q", "K"])
                else:
                    card_str = str(card)
                self.player_cards.append([suit, card_str])
                if is_bust(self.player):
                    # Player busts immediately and loses
                    self.terminated = True
                    self.reward = -1.0
                else:
                    # Game continues, turn passes to dealer
                    self.reward = 0.0
                    # If the dealer has not stood, change turns
                    if not self.dealer_stick:
                        self.current_turn = "dealer"
            else:  # stick
                self.player_stick = True
                if self.dealer_stick:
                    # Both have stuck -> resolve outcome
                    self.terminated = True
                    self.reward = self._final_result()
                else:
                    # Dealer still active -> pass turn
                    self.current_turn = "dealer"

        # ----- Dealer's turn -----
        else:  # self.current_turn == "dealer"
            if action:  # hit
                card = draw_card(self.np_random)
                self.dealer.append(card)
                suit = self.np_random.choice(suits)
                if card == 1:
                    card_str = "A"
                elif card == 10:
                    card_str = self.np_random.choice(["J", "Q", "K"])
                else:
                    card_str = str(card)
                self.dealer_cards.append([suit, card_str])
                if is_bust(self.dealer):
                    # Dealer busts -> player wins
                    self.terminated = True
                    self.reward = 1.0
                else:
                    # Game continues
                    self.reward = 0.0
                    # If the player has not stood, change turns
                    if not self.player_stick:
                        self.current_turn = "player"
            else:  # stick
                self.dealer_stick = True
                if self.player_stick:
                    # Both have stuck -> resolve outcome
                    self.terminated = True
                    self.reward = self._final_result()
                else:
                    # Player still active -> pass turn
                    self.current_turn = "player"

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {
            "current_turn": self.current_turn,
            "player_stick": self.player_stick,
            "dealer_stick": self.dealer_stick,
        }
        return obs, self.reward, self.terminated, False, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Reset the environment.

        Also resets the turn-based flags so that:
        - player starts first,
        - neither side has stuck at the beginning.
        """
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Reset sequential-game state
        self.player_stick = False
        self.dealer_stick = False

        # Get a random number between 0 and 1
        sample = uniform(0, 1)

        # If the number is between [0.5,1.0], player starts
        if sample >= 0.5:
            self.current_turn = "player"

        # If the number is between [0.0,0.5], dealer starts
        elif sample < 0.5:
            self.current_turn = "dealer"

        self.player_cards = []
        self.dealer_cards = []
        suits = ["C", "D", "H", "S"]

        for card in self.player:
            suit = self.np_random.choice(suits)
            if card == 1:
                card_str = "A"
            elif card == 10:
                card_str = self.np_random.choice(["J", "Q", "K"])
            else:
                card_str = str(card)
            self.player_cards.append([suit, card_str])
        for card in self.dealer:
            suit = self.np_random.choice(suits)
            if card == 1:
                card_str = "A"
            elif card == 10:
                card_str = self.np_random.choice(["J", "Q", "K"])
            else:
                card_str = str(card)
            self.dealer_cards.append([suit, card_str])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"current_turn": self.current_turn}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[toy_text]`")

        obs = self._get_obs()
        player_sum, dealer_card_value, usable_ace = obs[0][0], obs[0][1], obs[0][2]
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 4
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 24

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 15)

        if not self.terminated:
            turn = "Player" if self.current_turn == "player" else "Dealer"
            turn_text = small_font.render(f"Current Turn: {turn}", True, white)
            turn_text_rect = self.screen.blit(turn_text, (screen_width // 2 - turn_text.get_width() // 2, spacing))
        else:
            winner = None
            if self.reward == 1.0:
                winner = "Player"
            elif self.reward == -1.0:
                winner = "Dealer"
            if winner:
                turn_text = small_font.render(f"{winner} Wins", True, white)
                turn_text_rect = self.screen.blit(turn_text, (screen_width // 2 - turn_text.get_width() // 2, spacing))
            else:
                turn_text = small_font.render("Draw", True, white)
                turn_text_rect = self.screen.blit(turn_text, (screen_width // 2 - turn_text.get_width() // 2, spacing))

        if not self.terminated:
            dealer_text = small_font.render(f"Dealer: {dealer_card_value}", True, white)
        elif self.terminated:
            dealer_text = small_font.render(f"Dealer: {sum_hand(self.dealer)}", True, white)
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, turn_text_rect.bottom + spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_width = len(self.dealer_cards) * card_img_width + (len(self.dealer_cards) - 1) * spacing
        for card in self.dealer_cards:
            card_index = self.dealer_cards.index(card)
            if (card_index > 0 and self.terminated) or card_index == 0:
                dealer_card_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{card[0]}{card[1]}.png",
                        )
                    )
                )
                dealer_card_rect = self.screen.blit(
                    dealer_card_img,
                    (
                        (screen_width - dealer_width) // 2 + self.dealer_cards.index(card) * (card_img_width + spacing),
                        dealer_text_rect.bottom + spacing,
                    ),
                )
            else:
                hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
                self.screen.blit(
                    hidden_card_img,
                    (
                        (screen_width - dealer_width) // 2 + card_index * (card_img_width + spacing),
                        dealer_text_rect.bottom + spacing,
                    ),
                )

        player_text = small_font.render(f"Player: {player_sum}", True, white)
        player_text_rect = self.screen.blit(player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing))

        player_width = len(self.player_cards) * card_img_width + (len(self.player_cards) - 1) * spacing
        for card in self.player_cards:
            player_card_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{card[0]}{card[1]}.png",
                    )
                )
            )
            player_card_rect = self.screen.blit(
                player_card_img,
                (
                    (screen_width - player_width) // 2 + self.player_cards.index(card) * (card_img_width + spacing),
                    player_text_rect.bottom + spacing,
                ),
            )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_card_rect.bottom + card_img_height + spacing // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


# Pixel art from Mariia Khmelnytska
# (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
