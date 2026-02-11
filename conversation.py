#!/usr/bin/env python3
#
# Sign Language Chat — Conversation Module
#
# Manages vocabulary, message sequencing, and simple responses.
# Only uses the 24 static ASL letters (no J or Z).
#

import config


# Simple vocabulary — words that avoid J and Z
GREETINGS = ["HI", "HELLO", "HEY"]
RESPONSES = {
    "HI": "HELLO",
    "HELLO": "HI",
    "HEY": "HI",
    "HOW": "GOOD",
    "GOOD": "THANKS",
    "THANKS": "WELCOME",
    "WELCOME": "BYE",
    "BYE": "BYE",
    "YES": "OK",
    "NO": "OK",
    "OK": "COOL",
    "COOL": "NICE",
    "NICE": "THANKS",
    "WHAT": "NOTHING",
    "WHO": "ME",
    "NAME": "GRETCHEN",
}

# Scripted conversation for demo mode
DEMO_SCRIPT_A = ["HELLO", "HOW", "GOOD", "BYE"]
DEMO_SCRIPT_B = ["HI", "GOOD", "THANKS", "BYE"]


def validate_word(word):
    """Check if a word only contains valid ASL letters (no J or Z)."""
    return all(c in config.LETTERS for c in word.upper())


def get_response(received_word):
    """Generate a simple response to a received word.

    Falls back to echoing the word if no predefined response exists.
    """
    word = received_word.upper().strip()

    if word in RESPONSES:
        return RESPONSES[word]

    # Default: echo back
    if validate_word(word):
        return word

    # Filter out invalid characters
    filtered = "".join(c for c in word if c in config.LETTERS)
    return filtered if filtered else "OK"


class ConversationManager:
    """Manages the flow of a conversation between two robots."""

    def __init__(self, script=None):
        """
        Args:
            script: Optional list of words to send in order (demo mode).
                    If None, uses response-based conversation.
        """
        self.script = list(script) if script else None
        self.script_index = 0
        self.sent_words = []
        self.received_words = []

    def get_next_word(self, last_received=None):
        """Get the next word to send.

        Args:
            last_received: The last word received from the other side.

        Returns:
            Word to send, or None if conversation is over.
        """
        if self.script:
            if self.script_index < len(self.script):
                word = self.script[self.script_index]
                self.script_index += 1
                self.sent_words.append(word)
                return word
            return None  # Script exhausted

        # Response mode
        if last_received:
            response = get_response(last_received)
            self.sent_words.append(response)
            return response

        # First message — start with a greeting
        if not self.sent_words:
            word = GREETINGS[0]
            self.sent_words.append(word)
            return word

        return None

    def receive_word(self, word):
        """Record a received word."""
        self.received_words.append(word.upper())

    @property
    def is_done(self):
        """Check if the conversation script is exhausted."""
        if self.script:
            return self.script_index >= len(self.script)
        return False

    def get_history(self):
        """Return conversation history as list of (direction, word) tuples."""
        history = []
        si, ri = 0, 0
        # Interleave based on who went first
        while si < len(self.sent_words) or ri < len(self.received_words):
            if si < len(self.sent_words):
                history.append(("sent", self.sent_words[si]))
                si += 1
            if ri < len(self.received_words):
                history.append(("received", self.received_words[ri]))
                ri += 1
        return history
