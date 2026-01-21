import time
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, delay_seconds, cooldown_hours):
        self.delay = delay_seconds
        self.cooldown = timedelta(hours=cooldown_hours)
        self.blocked_until = None

    def wait(self):
        time.sleep(self.delay)

    def block(self):
        self.blocked_until = datetime.now() + self.cooldown

    def is_blocked(self):
        if not self.blocked_until:
            return False
        return datetime.now() < self.blocked_until
