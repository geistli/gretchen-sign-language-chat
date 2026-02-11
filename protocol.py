#!/usr/bin/env python3
#
# Sign Language Chat — Turn-Taking Protocol
#
# Visual state machine for coordinating two robots/laptops.
# Uses colored screen borders detected via camera.
#

from enum import Enum
import time
import config


class State(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"         # Showing letters with green border
    DONE_SPEAKING = "done"        # Red border — signaling turn is over
    LISTENING = "listening"       # Cyan border — ready to receive
    WAITING_FOR_TURN = "waiting"  # Watching for other side's red border


class TurnProtocol:
    """Manages the turn-taking state machine.

    The protocol works as follows:
    1. Speaker shows letters one by one with GREEN border
    2. When done, speaker shows RED border
    3. Speaker transitions to LISTENING (CYAN border)
    4. Listener detects RED → knows message is complete
    5. Listener becomes speaker, starts showing letters with GREEN border
    """

    def __init__(self, starts_as_speaker=True):
        if starts_as_speaker:
            self.state = State.SPEAKING
        else:
            self.state = State.LISTENING

        self._done_time = None
        self._done_duration = 2.0  # seconds to show red before switching

    @property
    def is_speaking(self):
        return self.state == State.SPEAKING

    @property
    def is_listening(self):
        return self.state == State.LISTENING

    @property
    def is_done_speaking(self):
        return self.state == State.DONE_SPEAKING

    @property
    def is_waiting(self):
        return self.state == State.WAITING_FOR_TURN

    def get_border_color(self):
        """Return the BGR border color for the current state."""
        if self.state == State.SPEAKING:
            return config.COLOR_GREEN
        elif self.state == State.DONE_SPEAKING:
            return config.COLOR_RED
        elif self.state == State.LISTENING:
            return config.COLOR_CYAN
        else:
            return config.COLOR_GRAY

    def finish_speaking(self):
        """Called when the speaker has finished sending all letters."""
        self.state = State.DONE_SPEAKING
        self._done_time = time.time()

    def update(self, detected_border_color=None):
        """Update state machine based on detected border color from camera.

        Args:
            detected_border_color: "green", "red", "cyan", or None

        Returns:
            Significant event string or None:
            - "turn_received": other side signaled done, we should start speaking
            - "done_timeout": we've shown red long enough, switch to listening
            - "letter_incoming": other side is showing a letter (green border)
        """
        if self.state == State.DONE_SPEAKING:
            # Wait a bit then switch to listening
            if self._done_time and time.time() - self._done_time > self._done_duration:
                self.state = State.LISTENING
                return "done_timeout"

        elif self.state == State.LISTENING:
            if detected_border_color == "green":
                return "letter_incoming"
            elif detected_border_color == "red":
                # Other side is done, our turn to speak
                self.state = State.SPEAKING
                return "turn_received"

        elif self.state == State.WAITING_FOR_TURN:
            if detected_border_color == "cyan":
                # Other side is ready to listen
                self.state = State.SPEAKING
                return "turn_received"

        return None

    def start_speaking(self):
        """Force transition to speaking state."""
        self.state = State.SPEAKING

    def start_listening(self):
        """Force transition to listening state."""
        self.state = State.LISTENING
